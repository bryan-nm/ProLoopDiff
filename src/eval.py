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

from config import CFG, CKPT_DIR, SWISSPROT_CSV
from .dist import init_distributed, broadcast_checkpoint_bytes, barrier, cleanup
from .recurrent_oadm import RecurrentOADM, count_params
from .sampler import generate
from .data import ProteinTokenizer, load_swissprot

try:
    import warnings as _w
    _w.filterwarnings("ignore", message=".*split master weight.*")
    import intel_extension_for_pytorch as ipex
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


def _pack_canvas(id_lists, pad_id, canvas_width):
    """Pack variable-length token lists into a (B, canvas_width) tensor, right-padded."""
    B = len(id_lists)
    canvas = torch.full((B, canvas_width), pad_id, dtype=torch.long)
    for i, ids in enumerate(id_lists):
        n = min(len(ids), canvas_width)
        canvas[i, :n] = torch.tensor(ids[:n], dtype=torch.long)
    return canvas


def lcr_baselines(sp_rows, cfg, n, canvas_width, env, dev, seed=42):
    """LCR on natural SwissProt sequences and their per-sequence shuffles.

    Each rank draws a distinct strided slice of SwissProt so the baselines cover more of the
    dataset. Statistics are all-reduced the same way as the generative eval.
    """
    torch.manual_seed(seed + env.rank)
    total = len(sp_rows)
    if total == 0:
        return None

    # each rank takes a strided slice, up to n sequences
    indices = list(range(env.rank, total, max(env.world_size, 1)))[:n]
    seqs = [sp_rows[i][1] for i in indices]  # ids lists (AA... EOS)
    if not seqs:
        return None

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
    return nat_frac, shuf_frac, n_seqs


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
    return mean, sd, int(stats[3].item()), n_total, lcr_frac


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
    args = ap.parse_args()

    env = init_distributed(args.device)
    dev = env.device
    mcfg, ocfg = CFG.model_config(), CFG.opt

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
    use_amp = dev.type in ("xpu", "cuda")
    ckpt_dir = args.ckpt_dir

    evaluated = set()

    if env.is_main:
        print(f"[eval] watching {ckpt_dir} (poll every {args.poll}s, "
              f"{ev_n} samples/rank x {env.world_size} ranks = {ev_n * env.world_size} total, "
              f"canvas={ev_canvas}, steps={ev_steps})", flush=True)

    # --- LCR baselines from natural sequences (before first checkpoint) ---
    tok = ProteinTokenizer(mcfg)
    sp_rows = load_swissprot(SWISSPROT_CSV, tok) if os.path.exists(SWISSPROT_CSV) else []
    if sp_rows:
        bl = lcr_baselines(sp_rows, mcfg, ev_n, ev_canvas, env, dev)
        if bl is not None and env.is_main:
            nat_frac, shuf_frac, n_bl = bl
            print(f"[eval] LCR baselines ({n_bl} SwissProt seqs): "
                  f"natural {nat_frac:.1%}, shuffled {shuf_frac:.1%}", flush=True)
        del sp_rows
    elif env.is_main:
        print(f"[eval] WARNING: {SWISSPROT_CSV} not found, skipping LCR baselines", flush=True)

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
            mean, sd, n_no_eos, n_total, lcr_frac = run_eval(
                model, dev, ev_n, ev_canvas, ev_steps, env, mcfg)
        _device_sync(dev)
        dt = time.perf_counter() - t0

        if env.is_main:
            print(f"[eval] step {ck_step} | len mean {mean:.1f} sd {sd:.1f} | "
                  f"no-EOS {n_no_eos}/{n_total} | LCR {lcr_frac:.1%} | {dt:.1f}s", flush=True)

        evaluated.add(step)
        time.sleep(args.poll)

    barrier()
    cleanup()


if __name__ == "__main__":
    main()
