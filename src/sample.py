"""Sample protein sequences UNCONDITIONALLY from a trained checkpoint.

Loads config.CKPT_DIR/latest.txt (or --ckpt), rebuilds the model, and runs the confidence-ordered
sampler with no text (text_emb=None -> the learned null condition), then writes FASTA. The canvas is a
single fixed length throughout decoding, so shapes stay static (no XPU recompilation).

Run (one tile is plenty):
    python -m src.sample --n 32 --out samples.fasta
    python -m src.sample --ckpt /path/ckpt_00050000.pt --n 16 --steps 128 --temperature 1.0
"""
from __future__ import annotations
import argparse
import os
import torch

from config import CFG, CKPT_DIR
from .dist import init_distributed
from .recurrent_oadm import RecurrentOADM
from .sampler import generate
from .train import find_latest_ckpt
from .blosum import AA


def decode(row, cfg) -> str:
    """Token row -> amino-acid string, stopping at EOS/PAD."""
    out = []
    for t in row.tolist():
        if t == cfg.eos_token_id or t == cfg.pad_token_id:
            break
        if 0 <= t < len(AA):
            out.append(AA[t])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt (default: latest in config.CKPT_DIR)")
    ap.add_argument("--n", type=int, default=16, help="number of sequences to sample")
    ap.add_argument("--canvas", type=int, default=512, help="max-length canvas (use a training bucket length)")
    ap.add_argument("--steps", type=int, default=None, help="decoding steps (default: canvas length; fewer = faster)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gumbel", type=float, default=0.1, help="stochastic-order noise (>0 adds diversity)")
    ap.add_argument("--corrector", type=int, default=0, help="remask corrector sweeps after decoding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="FASTA output path (default: stdout)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = init_distributed(args.device)                        # single process -> world=1, picks xpu/cpu
    dev = env.device
    cfg = CFG.model_config()
    model = RecurrentOADM(cfg).to(dev).eval()

    ckpt = args.ckpt or find_latest_ckpt(CKPT_DIR)
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint found (looked at: {args.ckpt or CKPT_DIR})")
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state["model"])
    print(f"[sample] loaded {ckpt} (step {state.get('step', '?')}) on {dev}; "
          f"sampling {args.n} sequences, canvas<={args.canvas}", flush=True)

    use_amp = dev.type in ("xpu", "cuda")
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
        canvas, lengths = generate(model, Lmax=args.canvas, batch_size=args.n, text_emb=None,
                                   cfg_weight=0.0, n_steps=args.steps, temperature=args.temperature,
                                   gumbel_temp=args.gumbel, n_corrector=args.corrector, device=str(dev))

    records = []
    for i in range(args.n):
        seq = decode(canvas[i], cfg)
        records.append(f">sample_{i} len={len(seq)}\n{seq}")
    text = "\n".join(records) + "\n"

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[sample] wrote {args.n} sequences -> {args.out}", flush=True)
    else:
        print(text)


if __name__ == "__main__":
    main()
