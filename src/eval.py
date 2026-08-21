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


def _decode_seqs(canvas, cfg, min_len):
    """Canvas rows -> amino-acid strings, truncated at the first EOS/PAD.

    Only ids 0..19 map to residues; anything else (a MASK that survived decoding) is dropped.
    Rows shorter than `min_len` are omitted -- ESMFold rejects empty input and a handful of
    residues carries no foldable signal. Returns (sequences, n_skipped).
    """
    seqs, skipped = [], 0
    for row in canvas.cpu().tolist():
        out = []
        for t in row:
            if t == cfg.eos_token_id or t == cfg.pad_token_id:
                break
            if 0 <= t < len(AA):
                out.append(AA[t])
        if len(out) >= min_len:
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

    def _vec(v):
        return torch.tensor(v, dtype=torch.float32, device=dev) if v else \
            torch.zeros(0, dtype=torch.float32, device=dev)

    p, t = _vec(plddt), _vec(ptm)
    stats = torch.zeros(6, device=dev)
    stats[0] = p.numel()
    stats[1] = p.sum()
    stats[2] = (p > ocfg.plddt_confident).sum()
    stats[3] = t.sum()
    stats[4] = (t > ocfg.ptm_confident).sum()
    stats[5] = n_skipped

    if env.distributed:
        torch.distributed.all_reduce(stats)

    n = int(stats[0].item())
    return {
        "plddt": 100.0 * stats[1].item() / max(n, 1),   # 0-1 scale -> AlphaFold 0-100
        "plddt_conf": stats[2].item() / max(n, 1),
        "ptm": stats[3].item() / max(n, 1),             # native 0-1
        "ptm_conf": stats[4].item() / max(n, 1),
        "n": n,
        "skipped": int(stats[5].item()),
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

    # Each rank takes a strided slice, up to n sequences. A rank whose slice comes out EMPTY
    # (fewer rows than ranks) must NOT return early -- it still has to enter the all_reduce below
    # or the ranks that do have work block there until the job hits its walltime.
    indices = list(range(env.rank, total, max(env.world_size, 1)))[:n]
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

    stats = torch.zeros(4, device=dev)
    stats[0] = nat_lcr
    stats[1] = nat_total
    stats[2] = shuf_lcr
    stats[3] = shuf_total

    if env.distributed:
        torch.distributed.all_reduce(stats)

    nat_frac = stats[0].item() / max(stats[1].item(), 1)
    shuf_frac = stats[2].item() / max(stats[3].item(), 1)
    n_seqs = len(indices)
    if env.distributed:
        n_buf = torch.tensor([n_seqs], dtype=torch.long, device=dev)
        torch.distributed.all_reduce(n_buf)
        n_seqs = int(n_buf.item())
    return nat_frac, shuf_frac, n_seqs, canvas, canvas_shuf


def run_eval(model, dev, n_per_rank, canvas, steps, env, cfg, seed=1234):
    """Sample unconditionally on every rank (distinct seeds), all-reduce the statistics."""
    torch.manual_seed(seed + env.rank)
    model.eval()
    tokens, lengths = generate(model, Lmax=canvas, batch_size=n_per_rank, text_emb=None, cfg_weight=0.0,
                               n_steps=steps, temperature=1.0, gumbel_temp=0.1, greedy=False,
                               device=str(dev))
    lens = torch.tensor(lengths, dtype=torch.float32, device=dev)
    lcr_res, total_res = _lcr_counts(tokens, cfg.eos_token_id)

    stats = torch.zeros(6, device=dev)
    stats[0] = len(lengths)
    stats[1] = lens.sum()
    stats[2] = (lens ** 2).sum()
    stats[3] = (lens >= canvas).sum()
    stats[4] = lcr_res
    stats[5] = total_res

    if env.distributed:
        torch.distributed.all_reduce(stats)

    n_total = int(stats[0].item())
    mean = stats[1].item() / max(n_total, 1)
    var = stats[2].item() / max(n_total, 1) - mean ** 2
    sd = var ** 0.5 if var > 0 else 0.0
    lcr_frac = stats[4].item() / max(stats[5].item(), 1)
    return mean, sd, int(stats[3].item()), n_total, lcr_frac, tokens


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
    args = ap.parse_args()

    env = init_distributed(args.device)
    dev = env.device
    mcfg, ocfg = CFG.model_config(), CFG.opt

    # The scorer logs model load + XPU patch details at INFO. Useful once, x12 ranks it is noise --
    # keep rank 0 (the patch lines are worth seeing after an esm/torch upgrade) and quiet the rest.
    # Only this logger's level is touched: calling basicConfig here would raise the ROOT level and
    # could switch ON other libraries' INFO output instead of reducing anything. A genuine load
    # failure is unaffected -- that is caught and printed explicitly below.
    if not env.is_main:
        import logging
        logging.getLogger("esmfold_scorer").setLevel(logging.WARNING)

    model = RecurrentOADM(mcfg).to(dev)
    model.eval()
    applied_ipex = ipex is not None and dev.type == "xpu" and ocfg.use_ipex and not args.no_ipex
    if applied_ipex:
        model = ipex.optimize(model, dtype=torch.bfloat16)
    if env.is_main:
        print(f"[eval] params={count_params(model)/1e6:.1f}M device={dev} "
              f"ipex={applied_ipex} world={env.world_size}", flush=True)

    ev_n = args.eval_n or ocfg.eval_n
    ev_canvas = args.eval_canvas or ocfg.eval_canvas
    ev_steps = args.eval_steps or ocfg.eval_steps
    fold_n = ocfg.fold_n if args.fold_n is None else args.fold_n
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
                                     num_loops=ocfg.fold_loops, num_diffusion_samples=1)
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
                for label, cv in (("natural", nat_canvas), ("shuffled", shuf_canvas)):
                    seqs, n_skip = _decode_seqs(cv[:fold_n], mcfg, ocfg.fold_min_len)
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
            time.sleep(args.poll)
            continue

        step = _parse_step(ckpt_path)
        if step in evaluated:
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
            seqs, n_skip = _decode_seqs(gen[:fold_n], mcfg, ocfg.fold_min_len)
            m = fold_stats(seqs, scorer, env, dev, ocfg, n_skip)
            if env.is_main:
                print(f"[eval] step {ck_step} | {_fold_line(m, ocfg)}", flush=True)
        del gen

        evaluated.add(step)
        time.sleep(args.poll)

    barrier()
    cleanup()


if __name__ == "__main__":
    main()
