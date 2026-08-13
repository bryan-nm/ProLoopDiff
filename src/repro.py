"""Reproduce and LOCALIZE a deterministic training GPU fault on a single tile.

The fault is deterministic (same rank, same step, different physical node), so the offending batch can
be reconstructed on ONE tile and replayed with a device-sync + print after each stage. Whatever line
prints LAST before the process aborts is the faulting stage. Run in a 1-node interactive session:

    python -m src.repro --rank 159 --world 192 --step 650 --device xpu
    python -m src.repro --rank 159 --world 192 --step 650 --device xpu --filip-cpu   # test the CPU-FiLIP fix

The batch CONTENT is exact (the sampler is seeded/deterministic); the corruption randomness is not
replayed, so we loop the step a few times -- if the fault is batch-driven it will still reproduce.
"""
from __future__ import annotations
import argparse
import os
import torch

from config import CFG, SWISSPROT_CSV, UNANNOTATED_SHARDS, TEXT_ENCODER, TEXT_CACHE, BLOSUM_MAT
from .dist import init_distributed
from .recurrent_oadm import RecurrentOADM
from .objective import sample_corruption, oadm_loss, filip_loss
from .blosum import blosum_sub_probs
from .data import (ProteinTokenizer, DummyTextEmbedder, HFTextEmbedder, CacheTextEmbedder, ProteinShards,
                   load_swissprot, MixedProteinDataset, BucketedLengthSampler, make_collate)


def _sync(dev):
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def staged_step(model, batch, cfg, ocfg, sub_probs, dev, filip_cpu, it):
    tokens, labelled = batch["tokens"], batch["labelled"]
    text_emb, text_keep = batch["text_emb"], batch["text_keep"]
    target_mask = torch.ones_like(tokens, dtype=torch.bool)
    cond_mask = labelled & (torch.rand(tokens.shape[0], device=dev) > ocfg.p_uncond)

    corrupted, mask_pos = sample_corruption(tokens, target_mask, cfg.mask_token_id,
                                            beta=ocfg.beta, sub_probs=sub_probs, beta_schedule=ocfg.beta_schedule)
    _sync(dev); print(f"[{it}] corruption OK", flush=True)

    logits, filip_pairs = model(corrupted, text_emb=text_emb, text_keep=text_keep,
                                cond_mask=cond_mask, collect_filip="last")
    _sync(dev); print(f"[{it}] forward OK", flush=True)

    L_oadm = oadm_loss(logits, tokens, mask_pos, pad_id=cfg.pad_token_id, pad_weight=ocfg.pad_loss_weight)
    _sync(dev); print(f"[{it}] oadm OK ({float(L_oadm):.3f})", flush=True)

    prot_keep = target_mask & (tokens != cfg.eos_token_id) & (tokens != cfg.pad_token_id)
    lab_idx = labelled.nonzero(as_tuple=True)[0]
    if lab_idx.numel() > ocfg.filip_max_rows:
        lab_idx = lab_idx[:ocfg.filip_max_rows]
    fdev = torch.device("cpu") if filip_cpu else dev
    li = lab_idx.to(fdev)
    pk, tk = prot_keep.to(fdev).index_select(0, li), text_keep.to(fdev).index_select(0, li)
    acc = 0.0
    for z_pre, zt_real in filip_pairs:
        acc = acc + filip_loss(z_pre.to(fdev).index_select(0, li), zt_real.to(fdev).index_select(0, li), pk, tk)
    L_filip = acc / len(filip_pairs)
    _sync(dev); print(f"[{it}] filip OK ({float(L_filip):.3f})", flush=True)

    (L_oadm + ocfg.lam_filip * L_filip.to(dev)).backward()
    _sync(dev); print(f"[{it}] backward OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--step", type=int, required=True, help="the crashing step for that rank")
    ap.add_argument("--iters", type=int, default=20, help="replays of the batch (corruption varies)")
    ap.add_argument("--filip-cpu", action="store_true")
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
    print(f"[repro] rank={args.rank} world={args.world} step={args.step} on {dev} | "
          f"batch tokens={tuple(batch['tokens'].shape)} n_labelled={int(batch['labelled'].sum())} "
          f"filip_cpu={args.filip_cpu}", flush=True)

    sub_probs = blosum_sub_probs(BLOSUM_MAT, temp=ocfg.blosum_temp).to(dev) if os.path.exists(BLOSUM_MAT) else None
    model = RecurrentOADM(cfg).to(dev)
    model.train()
    for it in range(args.iters):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type in ("xpu", "cuda")):
            staged_step(model, batch, cfg, ocfg, sub_probs, dev, args.filip_cpu, it)
    print("[repro] completed all iters WITHOUT a fault", flush=True)


if __name__ == "__main__":
    main()
