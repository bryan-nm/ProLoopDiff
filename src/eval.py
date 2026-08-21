"""Standalone checkpoint evaluator for the recurrent-OADM protein generator.

Watches CKPT_DIR for new checkpoints (via latest.txt) and evaluates each one as it appears.
Designed to run on a dedicated node alongside a training job: submit eval.pbs (1 node, 12 ranks)
at the same time as train.pbs (15 nodes, 180 ranks).

Each rank on the eval node generates samples with a distinct seed, then statistics are all-reduced
across ranks -- 12 tiles generating independently gives 12x the sample count at no extra wall time.

Usage:
    mpiexec -n 12 -ppn 12 python -m src.eval --device xpu
    python -m src.eval --device cpu --eval-n 8 --eval-steps 8    # local smoke test
"""
from __future__ import annotations
import io
import os
import re
import time
import argparse
import torch

from config import CFG, CKPT_DIR, SWISSPROT_CSV, ESMFOLD_WEIGHTS
from .dist import init_distributed, broadcast_checkpoint_bytes, barrier, cleanup
from .recurrent_oadm import RecurrentOADM, count_params
from .sampler import generate
from .data import ProteinTokenizer, load_swissprot
from .blosum import AA                      # canonical 20-AA alphabet, id == index

try:
    import intel_extension_for_pytorch as ipex
    import logging as _logging
    _logging.getLogger("IPEX").setLevel(_logging.WARNING)   # logger is named "IPEX", not the module
except Exception:
    ipex = None


def _find_latest_ckpt(ckpt_dir):
    p = os.path.join(ckpt_dir, "latest.txt")
    if not os.path.exists(p):
        return None
    full = os.path.join(ckpt_dir, open(p).read().strip())
    return full if os.path.exists(full) else None


def _parse_step(ckpt_path):
    m = re.search(r'ckpt_(\d+)\.pt', os.path.basename(ckpt_path))
    return int(m.group(1)) if m else -1


def _device_sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


# --- one persistent all-reduce buffer for EVERY collective this script runs -------------------
# Same hard-won reasoning as dist.preallocate_grad_buffer, which eval originally did not inherit.
# oneCCL caches L0 registrations keyed by POINTER. A stats tensor built per call is freed the
# moment the function returns, so CCL is left holding a registration for a block the allocator has
# taken back -- and ESMFold calls torch.xpu.empty_cache() between collectives, which returns such
# blocks to the DRIVER and unmaps them. The next collective then reads an unmapped page, which is
# exactly "Segmentation fault from GPU ... type: 0 (NotPresent), level: 0 (PTE)".
#
# One buffer, allocated once on a clean heap before ESMFold loads, reduced at a FIXED size every
# time: CCL sees exactly one (pointer, count) pair for the life of the run, and because the tensor
# stays live, empty_cache() can never unmap it.
_STATS_N = 16
_STATS_BUF = None


def preallocate_stats_buffer(dev):
    global _STATS_BUF
    _STATS_BUF = torch.zeros(_STATS_N, dtype=torch.float32, device=dev)
    return _STATS_BUF.numel()


def allreduce_stats(values, env, dev):
    """Sum `values` (a short list of scalars) across ranks; returns them as a list of floats."""
    global _STATS_BUF
    if len(values) > _STATS_N:
        raise ValueError(f"allreduce_stats takes at most {_STATS_N} values, got {len(values)}")
    if _STATS_BUF is None or _STATS_BUF.device != dev:
        preallocate_stats_buffer(dev)
    buf = _STATS_BUF
    buf.zero_()
    for i, v in enumerate(values):
        buf[i] = v
    if env.distributed:
        torch.distributed.all_reduce(buf)      # ALWAYS the full buffer: one fixed shape for CCL
    return buf[:len(values)].tolist()


def _lcr_counts(canvas, eos_id, window=12, threshold=2.2):
    """Count residue positions inside low-complexity windows (SEG-like sliding-window entropy).

    Slides a window of `window` residues across each sequence (up to the first EOS). Any window
    whose AA-composition Shannon entropy falls below `threshold` bits is flagged; a position is
    low-complexity if ANY overlapping window is flagged. Returns (lcr_positions, total_positions)
    as raw counts for cross-rank aggregation.

    SEG defaults for proteins: window=12, trigger threshold=2.2 bits. Max entropy for 20 AA
    types is log2(20) ~ 4.32; 2.2 bits corresponds to ~4.6 effective types, catching homopolymer
    runs and compositionally biased stretches like QQQNQQNQ.
    """
    canvas = canvas.cpu()
    B, L = canvas.shape
    total_res = 0
    lcr_res = 0

    for b in range(B):
        eos_pos = (canvas[b] == eos_id).nonzero(as_tuple=True)[0]
        seq_len = int(eos_pos.min()) if eos_pos.numel() > 0 else L
        total_res += seq_len
        if seq_len < window:
            continue

        seq = canvas[b, :seq_len]
        is_lcr = torch.zeros(seq_len, dtype=torch.bool)

        for i in range(seq_len - window + 1):
            counts = torch.bincount(seq[i:i + window], minlength=eos_id)[:eos_id]
            p = counts.float() / window
            p = p[p > 0]
            entropy = -(p * p.log2()).sum().item()
            if entropy < threshold:
                is_lcr[i:i + window] = True

        lcr_res += int(is_lcr.sum())

    return lcr_res, total_res


def _decode_seqs(canvas, cfg, min_len, max_len=None):
    """Canvas rows -> amino-acid strings, truncated at the first EOS/PAD.

    Only ids 0..19 map to residues; anything else (a MASK that survived decoding) is dropped.
    Rows outside [min_len, max_len] are OMITTED rather than clipped: ESMFold rejects empty input,
    a handful of residues carries no foldable signal, and silently folding a clipped fragment
    would report a pLDDT for a molecule that was never generated. Returns (sequences, n_skipped).
    """
    seqs, skipped = [], 0
    for row in canvas.cpu().tolist():
        out = []
        for t in row:
            if t == cfg.eos_token_id or t == cfg.pad_token_id:
                break
            if 0 <= t < len(AA):
                out.append(AA[t])
        if len(out) >= min_len and (max_len is None or len(out) <= max_len):
            seqs.append("".join(out))
        else:
            skipped += 1
    return seqs, skipped


def fold_stats(seqs, scorer, env, dev, ocfg, n_skipped=0):
    """Fold `seqs` with ESMFold2-Fast and all-reduce the pLDDT / pTM statistics.

    Every rank folds its OWN sequences concurrently on its own tile -- ESMFold2 scores one
    sequence at a time, so per-rank parallelism is the only parallelism available (see the
    EsmFold README). Both scores come back on a 0-1 scale; pLDDT is reported on AlphaFold's
    0-100 convention, pTM on its native 0-1. Returns a dict of the aggregated metrics.
    """
    t0 = time.perf_counter()
    plddt, ptm = [], []
    if seqs:                                    # a rank with nothing to fold still joins the collective
        res = scorer.score(seqs, num_sampling_steps=ocfg.fold_steps, num_loops=ocfg.fold_loops)
        plddt, ptm = res.per_sequence_plddt, res.per_sequence_ptm

    # Reduced on the HOST side: these are short Python lists, so there is no reason to put another
    # transient device tensor in front of oneCCL (see the allreduce_stats note above).
    n_p, sum_p = len(plddt), float(sum(plddt))
    conf_p = float(sum(1 for v in plddt if v > ocfg.plddt_confident))
    sum_t = float(sum(ptm))
    conf_t = float(sum(1 for v in ptm if v > ocfg.ptm_confident))

    n_f, sum_p, conf_p, sum_t, conf_t, skipped = allreduce_stats(
        [n_p, sum_p, conf_p, sum_t, conf_t, float(n_skipped)], env, dev)

    n = int(n_f)
    return {
        "plddt": 100.0 * sum_p / max(n, 1),             # 0-1 scale -> AlphaFold 0-100
        "plddt_conf": conf_p / max(n, 1),
        "ptm": sum_t / max(n, 1),                       # native 0-1
        "ptm_conf": conf_t / max(n, 1),
        "n": n,
        "skipped": int(skipped),
        "seconds": time.perf_counter() - t0,
    }


def _fold_line(m, ocfg):
    """One-line rendering of a fold_stats result."""
    return (f"pLDDT {m['plddt']:.1f} (>{ocfg.plddt_confident * 100:.0f}: {m['plddt_conf']:.1%}) | "
            f"pTM {m['ptm']:.3f} (>{ocfg.ptm_confident:.2f}: {m['ptm_conf']:.1%}) | "
            f"n={m['n']} skipped={m['skipped']} | {m['seconds']:.1f}s")


def _pack_canvas(id_lists, pad_id, canvas_width):
    """Pack variable-length token lists into a (B, canvas_width) tensor, right-padded."""
    B = len(id_lists)
    canvas = torch.full((B, canvas_width), pad_id, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        n = min(len(ids), canvas_width)
        canvas[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
    return canvas


def natural_baselines(sp_rows, cfg, n, canvas_width, env, dev, seed=42):
    """Build the natural / shuffled reference canvases and their LCR fractions.

    Natural SwissProt sequences are the best case; per-sequence shuffles preserve each
    sequence's exact AA composition while destroying its order, which is the right control for
    both metrics -- it isolates "is the ORDER meaningful" from "is the composition plausible".

    Each rank draws a distinct strided slice so the baselines cover more of the dataset.
    Returns (nat_lcr, shuf_lcr, n_seqs, nat_canvas, shuf_canvas); the canvases are handed to
    the folding pass so the sequences are decoded exactly once.
    """
    torch.manual_seed(seed + env.rank)
    total = len(sp_rows)
    if total == 0:
        return None

    # Only sequences that FIT the canvas. SwissProt is not length-filtered upstream -- the single
    # 512 bucket clips every length into itself -- so without this the baseline silently folds a
    # 512-residue FRAGMENT of each large protein and calls it "natural", which is both a
    # meaningless structure and an unfairly low reference for the model to be compared against.
    eligible = [i for i, r in enumerate(sp_rows) if len(r[1]) <= canvas_width]

    # Each rank takes a strided slice, up to n sequences. A rank whose slice comes out EMPTY
    # (fewer rows than ranks) must NOT return early -- it still has to enter the all_reduce below
    # or the ranks that do have work block there until the job hits its walltime.
    indices = eligible[env.rank::max(env.world_size, 1)][:n]
    seqs = [sp_rows[i][1] for i in indices]  # ids lists (AA... EOS)

    canvas = _pack_canvas(seqs, cfg.pad_token_id, canvas_width)
    nat_lcr, nat_total = _lcr_counts(canvas, cfg.eos_token_id)

    # per-sequence shuffle: permute the AA portion (before EOS), preserving composition
    shuffled_seqs = []
    for ids in seqs:
        eos_at = ids.index(cfg.eos_token_id) if cfg.eos_token_id in ids else len(ids)
        aa_part = list(ids[:eos_at])
        perm = torch.randperm(len(aa_part)).tolist()
        shuffled_seqs.append([aa_part[j] for j in perm] + ids[eos_at:])

    canvas_shuf = _pack_canvas(shuffled_seqs, cfg.pad_token_id, canvas_width)
    shuf_lcr, shuf_total = _lcr_counts(canvas_shuf, cfg.eos_token_id)

    # One collective for all five figures (the sequence count rode in its own all_reduce before,
    # which was a second transient buffer for CCL to cache and the allocator to recycle).
    nat_lcr, nat_total, shuf_lcr, shuf_total, n_seqs = allreduce_stats(
        [float(nat_lcr), float(nat_total), float(shuf_lcr), float(shuf_total),
         float(len(indices))], env, dev)

    nat_frac = nat_lcr / max(nat_total, 1)
    shuf_frac = shuf_lcr / max(shuf_total, 1)
    return nat_frac, shuf_frac, int(n_seqs), canvas, canvas_shuf


def run_eval(model, dev, n_per_rank, canvas, steps, env, cfg, seed=1234):
    """Sample unconditionally on every rank (distinct seeds), all-reduce the statistics."""
    torch.manual_seed(seed + env.rank)
    model.eval()
    tokens, lengths = generate(model, Lmax=canvas, batch_size=n_per_rank, text_emb=None, cfg_weight=0.0,
                               n_steps=steps, temperature=1.0, gumbel_temp=0.1, greedy=False,
                               device=str(dev))
    lcr_res, total_res = _lcr_counts(tokens, cfg.eos_token_id)
    n_l = len(lengths)
    s1 = float(sum(lengths))
    s2 = float(sum(v * v for v in lengths))
    n_full = float(sum(1 for v in lengths if v >= canvas))

    n_total, s1, s2, n_full, lcr_res, total_res = allreduce_stats(
        [float(n_l), s1, s2, n_full, float(lcr_res), float(total_res)], env, dev)

    n_total = int(n_total)
    mean = s1 / max(n_total, 1)
    var = s2 / max(n_total, 1) - mean ** 2
    sd = var ** 0.5 if var > 0 else 0.0
    lcr_frac = lcr_res / max(total_res, 1)
    return mean, sd, int(n_full), n_total, lcr_frac, tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--no-ipex", action="store_true", help="skip ipex.optimize")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR, help="checkpoint directory to watch")
    ap.add_argument("--poll", type=int, default=60, help="seconds between checkpoint polls")
    ap.add_argument("--eval-n", type=int, default=None,
                    help="samples per rank per eval (default: config eval_n)")
    ap.add_argument("--eval-canvas", type=int, default=None,
                    help="canvas width for sampling (default: config eval_canvas)")
    ap.add_argument("--eval-steps", type=int, default=None,
                    help="decoding steps (default: config eval_steps)")
    ap.add_argument("--fold-n", type=int, default=None,
                    help="sequences folded per rank per eval (default: config fold_n; 0 disables)")
    ap.add_argument("--esmfold-weights", default=ESMFOLD_WEIGHTS,
                    help="ESMFold2-Fast weights directory or HF hub id")
    ap.add_argument("--fold-max-len", type=int, default=None,
                    help="skip sequences longer than this when folding (default: config fold_max_len)")
    ap.add_argument("--baselines", default="natural,shuffled",
                    help="which fold baselines to run, comma-separated ('natural', 'shuffled', or "
                         "'none'). Running 'shuffled' ALONE is the test that separates a "
                         "degenerate-input fault from one caused by state accumulated over a "
                         "preceding batch: if shuffled faults as the FIRST fold, the input is "
                         "the trigger.")
    ap.add_argument("--once", action="store_true",
                    help="run the baselines, evaluate the latest checkpoint if one exists, then "
                         "EXIT instead of watching. For debug-queue bisection runs, which would "
                         "otherwise sit in the poll loop until walltime.")
    args = ap.parse_args()

    env = init_distributed(args.device)
    dev = env.device
    mcfg, ocfg = CFG.model_config(), CFG.opt

    # Keep the scorer's INFO (model load, XPU patch lines) and esm's kernel-fallback WARNINGs on
    # rank 0 only -- they are worth reading once after an esm/torch upgrade, but x12 ranks they
    # bury the actual eval output. The esm config FutureWarnings are Python warnings, not logging,
    # so they need the warnings filter as well. Only these named loggers are touched: calling
    # basicConfig would raise the ROOT level and could switch ON other libraries' INFO instead of
    # reducing anything. A genuine scorer load failure is unaffected -- that is caught and printed
    # explicitly below, on every rank.
    if not env.is_main:
        import logging
        import warnings
        logging.getLogger("esmfold_scorer").setLevel(logging.WARNING)
        logging.getLogger("esm").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"esm\..*")
        warnings.filterwarnings("ignore", message=".*Disabling autocast.*")

    # BEFORE anything else touches the allocator, so CCL's one cached registration is for a block
    # taken from a clean heap and held for the life of the run. See allreduce_stats.
    preallocate_stats_buffer(dev)

    model = RecurrentOADM(mcfg).to(dev)
    model.eval()
    applied_ipex = ipex is not None and dev.type == "xpu" and ocfg.use_ipex and not args.no_ipex
    if applied_ipex:
        model = ipex.optimize(model, dtype=torch.bfloat16)
    if env.is_main:
        print(f"[eval] params={count_params(model)/1e6:.1f}M device={dev} "
              f"ipex={applied_ipex} world={env.world_size}", flush=True)
        # Versions, so an eval log can be diffed against EsmFold's speed_test banner: that test is
        # the known-good reference, and it ran in a DIFFERENT venv (esmfold-env). A version skew
        # here is a first thing to rule out when folding behaves differently than it did there.
        def _ver(mod):
            try:
                return __import__(mod).__version__
            except Exception:
                return "n/a"
        print(f"[eval] torch={torch.__version__} ipex={_ver('intel_extension_for_pytorch')} "
              f"transformers={_ver('transformers')} esm={_ver('esm')}", flush=True)

    ev_n = args.eval_n or ocfg.eval_n
    ev_canvas = args.eval_canvas or ocfg.eval_canvas
    ev_steps = args.eval_steps or ocfg.eval_steps
    fold_n = ocfg.fold_n if args.fold_n is None else args.fold_n
    fold_max_len = args.fold_max_len or ocfg.fold_max_len
    use_amp = dev.type in ("xpu", "cuda")
    ckpt_dir = args.ckpt_dir

    evaluated = set()

    if env.is_main:
        print(f"[eval] watching {ckpt_dir} (poll every {args.poll}s, "
              f"{ev_n} samples/rank x {env.world_size} ranks = {ev_n * env.world_size} total, "
              f"canvas={ev_canvas}, steps={ev_steps})", flush=True)

    # --- structural scorer (ESMFold2-Fast). Built ONCE: ~30s load, ~12.3GiB resident. ---
    # NOTE: on XPU the scorer installs process-global monkey-patches on torch.linalg.svd/det and
    # F.linear/F.layer_norm (see the EsmFold README). They are no-ops outside the cases they
    # target, and this is the eval process only -- training runs under a separate mpiexec launch.
    scorer, load_err = None, None
    if fold_n > 0:
        try:
            from esmfold_scorer import StructureScorer
            t_load = time.perf_counter()
            # Pass the BARE device type ("xpu"), not str(dev) ("xpu:0"): the scorer's
            # resolve_device() only accepts bare names and raises ValueError on an indexed one.
            # An unindexed torch.device resolves to the CURRENT device, which dist._pick_device
            # already pinned to this rank's tile via torch.xpu.set_device -- re-asserted here so
            # each of the 12 ranks loads its own 12.3GiB copy onto its own tile rather than all
            # piling onto tile 0.
            if dev.type == "xpu" and dev.index is not None:
                torch.xpu.set_device(dev.index)
            elif dev.type == "cuda" and dev.index is not None:
                torch.cuda.set_device(dev.index)
            scorer = StructureScorer(args.esmfold_weights, device=dev.type,
                                     num_sampling_steps=ocfg.fold_steps,
                                     num_loops=ocfg.fold_loops, num_diffusion_samples=1,
                                     empty_cache_every=ocfg.fold_empty_cache_every)
            # EVERY rank reports its own resident footprint, once. The scorer is handed a bare
            # "xpu" device, so it lands on whatever the CURRENT device is -- if that pinning ever
            # breaks, all 12 ranks load their own 12.3GiB copy onto tile 0 and the node dies with a
            # GPU fault that looks nothing like an OOM. ~12GiB on each rank's OWN index proves the
            # pinning held; 12 lines once per run is cheap next to debugging that from a page fault.
            resident = (torch.xpu.memory_allocated(dev) / 1024**3 if dev.type == "xpu" else
                        torch.cuda.memory_allocated(dev) / 1024**3 if dev.type == "cuda" else 0.0)
            print(f"[eval] rank {env.rank:>3} loaded ESMFold on {dev} | {resident:.1f}GiB resident",
                  flush=True)
            if env.is_main:
                print(f"[eval] ESMFold2-Fast loaded in {time.perf_counter() - t_load:.1f}s from "
                      f"{args.esmfold_weights} | folding {fold_n}/rank x {env.world_size} = "
                      f"{fold_n * env.world_size} seqs/round at {ocfg.fold_steps} steps", flush=True)
        except Exception as ex:
            scorer, load_err = None, f"{type(ex).__name__}: {ex}"
            print(f"[eval] rank {env.rank}: ESMFold scorer failed to load ({load_err})", flush=True)

        # Make the decision UNANIMOUS. Folding is collective (fold_stats all-reduces), so a rank
        # that quietly skipped it would leave every other rank blocked in that collective until
        # the job hit its walltime. If ANY rank failed to load, all of them disable folding.
        if env.distributed:
            ok = torch.tensor([1.0 if scorer is not None else 0.0], device=dev)
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
            if ok.item() == 0.0 and scorer is not None:
                scorer = None
        if scorer is None and env.is_main:
            print(f"[eval] WARNING: ESMFold unavailable on at least one rank; "
                  f"pLDDT reporting disabled for the whole run.", flush=True)
    elif env.is_main:
        print("[eval] folding disabled (fold_n=0); pLDDT will not be reported", flush=True)

    # --- baselines from natural sequences (before the first checkpoint lands) ---
    # Natural = best case, shuffled = composition-matched worst case. Both metrics get both
    # references, so every later number has a scale to be read against.
    tok = ProteinTokenizer(mcfg)
    sp_rows = load_swissprot(SWISSPROT_CSV, tok) if os.path.exists(SWISSPROT_CSV) else []
    if sp_rows:
        bl = natural_baselines(sp_rows, mcfg, max(ev_n, fold_n), ev_canvas, env, dev)
        del sp_rows
        if bl is not None:
            nat_frac, shuf_frac, n_bl, nat_canvas, shuf_canvas = bl
            if env.is_main:
                print(f"[eval] LCR baselines ({n_bl} SwissProt seqs): "
                      f"natural {nat_frac:.1%}, shuffled {shuf_frac:.1%}", flush=True)
            if scorer is not None:
                want = {s.strip() for s in args.baselines.split(",")} - {"", "none"}
                for label, cv in (("natural", nat_canvas), ("shuffled", shuf_canvas)):
                    if label not in want:
                        continue
                    seqs, n_skip = _decode_seqs(cv[:fold_n], mcfg, ocfg.fold_min_len, fold_max_len)
                    m = fold_stats(seqs, scorer, env, dev, ocfg, n_skip)
                    if env.is_main:
                        print(f"[eval] fold baseline {label:>8} | {_fold_line(m, ocfg)}", flush=True)
    elif env.is_main:
        print(f"[eval] WARNING: {SWISSPROT_CSV} not found, skipping baselines", flush=True)

    while True:
        ckpt_path = _find_latest_ckpt(ckpt_dir)
        if ckpt_path is None:
            if env.is_main and not evaluated:
                print(f"[eval] waiting for first checkpoint in {ckpt_dir}...", flush=True)
            if args.once:
                break
            time.sleep(args.poll)
            continue

        step = _parse_step(ckpt_path)
        if step in evaluated:
            if args.once:
                break
            time.sleep(args.poll)
            continue

        if env.is_main:
            print(f"[eval] loading step {step}: {os.path.basename(ckpt_path)}", flush=True)

        raw = broadcast_checkpoint_bytes(ckpt_path if env.is_main else None, dev)
        ck = torch.load(io.BytesIO(raw), map_location=dev, weights_only=True)
        del raw
        model.load_state_dict(ck["model"])
        ck_step = ck.get("step", step)
        del ck

        t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            mean, sd, n_no_eos, n_total, lcr_frac, gen = run_eval(
                model, dev, ev_n, ev_canvas, ev_steps, env, mcfg)
        _device_sync(dev)
        dt = time.perf_counter() - t0

        if env.is_main:
            print(f"[eval] step {ck_step} | len mean {mean:.1f} sd {sd:.1f} | "
                  f"no-EOS {n_no_eos}/{n_total} | LCR {lcr_frac:.1%} | {dt:.1f}s", flush=True)

        # Fold OUTSIDE the bf16 autocast: the scorer manages its own dtypes (and on XPU patches
        # F.linear to match weights), so an enclosing autocast would fight it.
        if scorer is not None:
            seqs, n_skip = _decode_seqs(gen[:fold_n], mcfg, ocfg.fold_min_len, fold_max_len)
            m = fold_stats(seqs, scorer, env, dev, ocfg, n_skip)
            if env.is_main:
                print(f"[eval] step {ck_step} | {_fold_line(m, ocfg)}", flush=True)
        del gen

        evaluated.add(step)
        if args.once:
            break
        time.sleep(args.poll)

    if env.is_main:
        print(f"[eval] done ({len(evaluated)} checkpoint(s) evaluated)", flush=True)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
