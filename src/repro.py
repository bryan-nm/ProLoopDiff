"""Reproduce and LOCALIZE a training GPU fault on a single tile.

The sampler is seeded and deterministic, so a given (rank, world, step) reconstructs the EXACT batch
that rank was running. Replay it here with a device-sync + print after each stage: whatever line
prints LAST before the process aborts is the faulting stage. Run in a 1-node interactive session:

    python -m src.repro --rank 142 --world 192 --step 5700 --device xpu
    python -m src.repro --rank 142 --world 192 --step 5700 --device xpu --no-ipex   # isolate ipex.optimize

The batch CONTENT is exact; the corruption randomness is not replayed, so we loop the step a few
times -- if the fault is batch-driven it will still reproduce.
"""
from __future__ import annotations
import argparse
import os
import torch

from config import CFG, SWISSPROT_CSV, UNANNOTATED_SHARDS, TEXT_ENCODER, TEXT_CACHE, BLOSUM_MAT
from .dist import init_distributed
from .recurrent_oadm import RecurrentOADM
from .objective import sample_corruption, oadm_loss
from .blosum import blosum_sub_probs
from .data import (ProteinTokenizer, DummyTextEmbedder, HFTextEmbedder, CacheTextEmbedder, ProteinShards,
                   load_swissprot, MixedProteinDataset, BucketedLengthSampler, make_collate)

try:
    import intel_extension_for_pytorch as ipex
except Exception:
    ipex = None


def _sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def staged_step(model, batch, cfg, ocfg, sub_probs, dev, it):
    tokens, labelled = batch["tokens"], batch["labelled"]
    text_emb, text_keep = batch["text_emb"], batch["text_keep"]
    target_mask = torch.ones_like(tokens, dtype=torch.bool)
    cond_mask = labelled & (torch.rand(tokens.shape[0], device=dev) > ocfg.p_uncond)

    corrupted, mask_pos = sample_corruption(tokens, target_mask, cfg.mask_token_id,
                                            beta=ocfg.beta, sub_probs=sub_probs, beta_schedule=ocfg.beta_schedule)
    _sync(dev); print(f"[{it}] corruption OK", flush=True)

    logits = model(corrupted, text_emb=text_emb, text_keep=text_keep, cond_mask=cond_mask)
    _sync(dev); print(f"[{it}] forward OK", flush=True)

    L_oadm = oadm_loss(logits, tokens, mask_pos, pad_id=cfg.pad_token_id, pad_weight=ocfg.pad_loss_weight)
    _sync(dev); print(f"[{it}] oadm OK ({float(L_oadm):.3f})", flush=True)

    L_oadm.backward()
    _sync(dev); print(f"[{it}] backward OK", flush=True)

    torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg.grad_clip)
    _sync(dev); print(f"[{it}] clip OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--step", type=int, required=True, help="the crashing step for that rank")
    ap.add_argument("--iters", type=int, default=20, help="replays of the batch (corruption varies)")
    ap.add_argument("--no-ipex", action="store_true", help="skip ipex.optimize (eager XPU)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    env = init_distributed(args.device)          # single process
    dev = env.device
    cfg, dcfg, ocfg = CFG.model_config(), CFG.data, CFG.opt
    torch.manual_seed(ocfg.seed + args.rank)

    tok = ProteinTokenizer(cfg)
    sp_rows = load_swissprot(SWISSPROT_CSV, tok)
    unannot = ProteinShards(UNANNOTATED_SHARDS, cfg.eos_token_id)
    ds = MixedProteinDataset(sp_rows, unannot if len(unannot) else None, dcfg.p_swissprot)
    embedder = (CacheTextEmbedder(TEXT_CACHE) if os.path.exists(os.path.join(TEXT_CACHE, "fingerprint.json"))
                else HFTextEmbedder(TEXT_ENCODER, dcfg.max_text_tokens) if os.path.isdir(TEXT_ENCODER)
                else DummyTextEmbedder(cfg.text_dim, dcfg.max_text_tokens))
    sampler = BucketedLengthSampler(ds.lengths, ds.n_sp_target, dcfg.length_buckets,
                                    ocfg.global_batch_tokens, dcfg.p_swissprot,
                                    rank=args.rank, world=args.world, seed=ocfg.seed)
    collate = make_collate(embedder, cfg, dcfg.length_buckets, dcfg.max_text_tokens)

    idxs = None                                  # fast-forward to this rank's `step`-th batch
    for i, b in enumerate(sampler):
        if i == args.step:
            idxs = b
            break
    if idxs is None:
        raise SystemExit(f"rank {args.rank} has fewer than {args.step} batches")
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([ds[j] for j in idxs]).items()}
    use_ipex = ipex is not None and dev.type == "xpu" and ocfg.use_ipex and not args.no_ipex
    print(f"[repro] rank={args.rank} world={args.world} step={args.step} on {dev} | "
          f"batch tokens={tuple(batch['tokens'].shape)} n_labelled={int(batch['labelled'].sum())} "
          f"ipex={use_ipex}", flush=True)

    sub_probs = blosum_sub_probs(BLOSUM_MAT, temp=ocfg.blosum_temp).to(dev) if os.path.exists(BLOSUM_MAT) else None
    model = RecurrentOADM(cfg).to(dev)
    # Mirror the trainer: ipex.optimize is part of what we are trying to implicate or clear.
    opt = torch.optim.AdamW(model.parameters(), lr=ocfg.lr, weight_decay=ocfg.weight_decay, betas=(0.9, 0.98))
    if use_ipex:
        model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16)
    model.train()
    for it in range(args.iters):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type in ("xpu", "cuda")):
            staged_step(model, batch, cfg, ocfg, sub_probs, dev, it)
    print("[repro] completed all iters WITHOUT a fault", flush=True)


if __name__ == "__main__":
    main()
