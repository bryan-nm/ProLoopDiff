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

from config import CFG, CKPT_DIR
from .dist import init_distributed, broadcast_checkpoint_bytes, barrier, cleanup
from .recurrent_oadm import RecurrentOADM, count_params
from .sampler import generate

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


def eval_lengths(model, dev, n_per_rank, canvas, steps, env, seed=1234):
    """Sample unconditionally on every rank (distinct seeds), all-reduce the statistics."""
    torch.manual_seed(seed + env.rank)
    model.eval()
    _, lengths = generate(model, Lmax=canvas, batch_size=n_per_rank, text_emb=None, cfg_weight=0.0,
                          n_steps=steps, temperature=1.0, gumbel_temp=0.1, greedy=False,
                          device=str(dev))
    lens = torch.tensor(lengths, dtype=torch.float32, device=dev)

    stats = torch.zeros(4, device=dev)
    stats[0] = len(lengths)
    stats[1] = lens.sum()
    stats[2] = (lens ** 2).sum()
    stats[3] = (lens >= canvas).sum()

    if env.distributed:
        torch.distributed.all_reduce(stats)

    n_total = int(stats[0].item())
    mean = stats[1].item() / max(n_total, 1)
    var = stats[2].item() / max(n_total, 1) - mean ** 2
    sd = var ** 0.5 if var > 0 else 0.0
    return mean, sd, int(stats[3].item()), n_total


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
            mean, sd, n_no_eos, n_total = eval_lengths(
                model, dev, ev_n, ev_canvas, ev_steps, env)
        _device_sync(dev)
        dt = time.perf_counter() - t0

        if env.is_main:
            print(f"[eval] step {ck_step} | len mean {mean:.1f} sd {sd:.1f} | "
                  f"no-EOS {n_no_eos}/{n_total} | {dt:.1f}s", flush=True)

        evaluated.add(step)
        time.sleep(args.poll)

    barrier()
    cleanup()


if __name__ == "__main__":
    main()
