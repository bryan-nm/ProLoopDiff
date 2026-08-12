"""Training entry point for the recurrent-OADM protein generator (Aurora XPU / oneCCL).

Wires config -> dist -> data -> model -> objective. Manual data-parallel (broadcast + coalesced
grad all-reduce, mini-embed convention) rather than DDP: the conditioning pathway always fires via
the learned null token, so the param-grad set is identical across ranks every step (collective-safe).

Launch (Aurora): via train.pbs (mpiexec, one rank per tile). Local smoke:
    PROTGEN_SWISSPROT_CSV=... python train.py --smoke
"""
from __future__ import annotations
import os
import math
import argparse
import torch

from config import CFG, SWISSPROT_CSV, UNANNOTATED_SHARDS, TEXT_ENCODER, TEXT_CACHE, BLOSUM_MAT, CKPT_DIR
from .dist import init_distributed, barrier, cleanup, broadcast_parameters, average_gradients
from .recurrent_oadm import RecurrentOADM, count_params
from .objective import training_step
from .blosum import blosum_sub_probs
from .data import (ProteinTokenizer, DummyTextEmbedder, HFTextEmbedder, CacheTextEmbedder, ProteinShards,
                   load_swissprot, MixedProteinDataset, BucketedBatchSampler, make_collate)

try:
    import intel_extension_for_pytorch as ipex
except Exception:
    ipex = None


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: few CSV rows, dummy text, a few steps")
    args = ap.parse_args()

    env = init_distributed(args.device)
    dev = env.device
    torch.manual_seed(CFG.opt.seed + env.rank)
    mcfg, dcfg, ocfg = CFG.model_config(), CFG.data, CFG.opt

    # --- model ---
    model = RecurrentOADM(mcfg).to(dev)
    if env.is_main:
        print(f"[train] params={count_params(model)/1e6:.1f}M device={dev}", flush=True)
    broadcast_parameters(model)
    sub_probs = blosum_sub_probs(BLOSUM_MAT, temp=ocfg.blosum_temp).to(dev) \
        if os.path.exists(BLOSUM_MAT) else None

    # --- data ---
    tok = ProteinTokenizer(mcfg)
    sp_rows = load_swissprot(SWISSPROT_CSV, tok, limit=256 if args.smoke else None) \
        if os.path.exists(SWISSPROT_CSV) else []
    unannot = ProteinShards(UNANNOTATED_SHARDS, mcfg.eos_token_id)
    ds = MixedProteinDataset(sp_rows, unannot if len(unannot) else None, dcfg.p_swissprot)

    # Text embedder: precomputed cache (preferred) -> live BERT -> dummy. Cache row order MUST match
    # load_swissprot order; verify against pair_ids.json before trusting text_idx alignment.
    has_cache = os.path.exists(os.path.join(TEXT_CACHE, "fingerprint.json"))
    if has_cache:
        embedder = CacheTextEmbedder(TEXT_CACHE)
        if not args.smoke and len(embedder) != len(sp_rows) and env.is_main:
            raise RuntimeError(f"text cache has {len(embedder)} rows but SwissProt has {len(sp_rows)}; "
                               f"re-run precompute_text (order must match load_swissprot).")
    else:
        use_dummy = args.smoke or not os.path.isdir(TEXT_ENCODER)
        if use_dummy and not args.smoke and env.is_main:
            print(f"[train] WARNING: no text cache and TEXT_ENCODER {TEXT_ENCODER!r} not found -> DUMMY "
                  f"embeddings; text/FiLIP learns nothing. Run src.precompute_text or set PROTGEN_TEXT_ENCODER.",
                  flush=True)
        embedder = DummyTextEmbedder(mcfg.text_dim, dcfg.max_text_tokens) if use_dummy \
            else HFTextEmbedder(TEXT_ENCODER, dcfg.max_text_tokens, device="cpu")

    mb = 4 if args.smoke else ocfg.micro_batch
    sampler = BucketedBatchSampler([len(ds[i]["ids"]) for i in range(len(ds))], mb,
                                   rank=env.rank, world=env.world_size, seed=ocfg.seed)
    # A live HF encoder holds a torch model that can't be forked to workers; the cache reader is fork-safe.
    nw = 0 if (args.smoke or isinstance(embedder, HFTextEmbedder)) else dcfg.num_workers
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=make_collate(embedder, mcfg),
                                         num_workers=nw)
    if env.is_main:
        print(f"[train] swissprot={len(sp_rows)} unannotated={len(unannot)} mixed={len(ds)} "
              f"text={'cache' if has_cache else type(embedder).__name__} batches/epoch/rank={len(sampler)}", flush=True)

    # --- optimizer / ipex ---
    opt = torch.optim.AdamW(model.parameters(), lr=ocfg.lr, weight_decay=ocfg.weight_decay, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, ocfg.warmup_steps, ocfg.total_steps))
    if ipex is not None and dev.type == "xpu":
        model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16)

    total = 60 if args.smoke else ocfg.total_steps
    log_every = 10 if args.smoke else ocfg.log_every
    use_amp = dev.type in ("xpu", "cuda")
    step = 0
    model.train()
    while step < total:
        sampler.set_epoch(step)
        for batch in loader:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, m = training_step(model, batch, p_uncond=ocfg.p_uncond, lam_filip=ocfg.lam_filip,
                                        beta=ocfg.beta, sub_probs=sub_probs, beta_schedule=ocfg.beta_schedule)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg.grad_clip)
            average_gradients(model)
            opt.step()
            sched.step()
            if env.is_main and step % log_every == 0:
                print(f"step {step:>7} | loss {m['total']:.3f} | oadm {m['oadm']:.3f} | filip {m['filip']:.3f} "
                      f"| cond {m['n_cond']}/{m['n_labelled']} | lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if env.is_main and step > 0 and step % ocfg.ckpt_every == 0:
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save({"model": model.state_dict(), "step": step}, os.path.join(CKPT_DIR, f"ckpt_{step}.pt"))
            step += 1
            if step >= total:
                break
    if env.is_main:
        print(f"[train] done at step {step}", flush=True)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
