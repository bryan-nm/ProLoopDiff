"""Training entry point for the recurrent-OADM protein generator (Aurora XPU / oneCCL).

Wires config -> dist -> data -> model -> objective. Manual data-parallel (broadcast + coalesced
grad all-reduce, mini-embed convention) rather than DDP: the conditioning pathway always fires via
the learned null token, so the param-grad set is identical across ranks every step (collective-safe).

Launch (Aurora): via train.pbs (mpiexec, one rank per tile). Local smoke:
    PROTGEN_SWISSPROT_CSV=... python train.py --smoke
"""
from __future__ import annotations
import io
import os
import math
import time
import argparse
import torch

from config import CFG, SWISSPROT_CSV, UNANNOTATED_SHARDS, TEXT_ENCODER, TEXT_CACHE, BLOSUM_MAT, CKPT_DIR
from .dist import (init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
                   broadcast_checkpoint_bytes, preallocate_grad_buffer)
from .recurrent_oadm import RecurrentOADM, count_params
from .objective import training_step
from .blosum import blosum_sub_probs
from .data import (ProteinTokenizer, DummyTextEmbedder, HFTextEmbedder, CacheTextEmbedder, ProteinShards,
                   load_swissprot, MixedProteinDataset, BucketedLengthSampler, make_collate)

try:
    import intel_extension_for_pytorch as ipex
    # IPEX logs "split master weight ... only support sgd" and two BatchNorm-folding lines at INFO
    # on every ipex.optimize call, on every rank. Its logger is named "IPEX" (see the
    # "_logger.py - IPEX - INFO" prefix in job output) -- NOT "intel_extension_for_pytorch".
    import logging as _logging
    _logging.getLogger("IPEX").setLevel(_logging.WARNING)
except Exception:
    ipex = None


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


# --- checkpointing (rank 0 writes; atomic via tmp+rename; latest.txt is the resume pointer) ---
def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(model, opt, sched, step, ckpt_dir, env, keep_last=3):
    if not env.is_main:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{step:08d}.pt")
    _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "step": step}, path)
    tmp = os.path.join(ckpt_dir, "latest.txt.tmp")     # update the pointer only after the ckpt is fully written
    with open(tmp, "w") as f:
        f.write(os.path.basename(path))
    os.replace(tmp, os.path.join(ckpt_dir, "latest.txt"))
    print(f"[ckpt] saved {path}", flush=True)
    if keep_last:                                      # rotate: keep only the newest `keep_last` (model+opt is big)
        import glob
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
        for old in ckpts[:-keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass


def find_latest_ckpt(ckpt_dir):
    p = os.path.join(ckpt_dir, "latest.txt")
    if not os.path.exists(p):
        return None
    full = os.path.join(ckpt_dir, open(p).read().strip())
    return full if os.path.exists(full) else None


def _device_sync(dev):
    # XPU/CUDA ops are async; sync before timing so tok/s reflects real compute, not enqueue time.
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def _peak_mem_gb(dev):
    # Cumulative high-water mark of allocated HBM -> the number that matters for OOM headroom.
    if dev.type == "xpu":
        return torch.xpu.max_memory_allocated() / 1e9
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: few CSV rows, dummy text, a few steps")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint and start from step 0")
    ap.add_argument("--max-steps", type=int, default=None, help="override total_steps (e.g. debug-queue validation)")
    ap.add_argument("--no-ipex", action="store_true", help="skip ipex.optimize (eager XPU; robust to dynamic shapes)")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None,
                    help="override grad_checkpoint (default: model config, currently True)")
    args = ap.parse_args()

    env = init_distributed(args.device)
    dev = env.device
    torch.manual_seed(CFG.opt.seed + env.rank)
    mcfg, dcfg, ocfg = CFG.model_config(), CFG.data, CFG.opt
    if args.grad_checkpoint is not None:
        mcfg.grad_checkpoint = args.grad_checkpoint

    # --- model ---
    model = RecurrentOADM(mcfg).to(dev)
    if env.is_main:
        print(f"[train] params={count_params(model)/1e6:.1f}M device={dev} "
              f"grad_checkpoint={mcfg.grad_checkpoint}", flush=True)
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
        # Every rank raises: failing on rank 0 alone leaves the other ranks blocked in a collective
        # until the job hits its walltime, with the real error buried on one rank's stream.
        if not args.smoke and len(embedder) != len(sp_rows):
            raise RuntimeError(f"text cache has {len(embedder)} rows but SwissProt has {len(sp_rows)}; "
                               f"re-run precompute_text (order must match load_swissprot).")
    else:
        use_dummy = args.smoke or not os.path.isdir(TEXT_ENCODER)
        if use_dummy and not args.smoke and env.is_main:
            print(f"[train] WARNING: no text cache and TEXT_ENCODER {TEXT_ENCODER!r} not found -> DUMMY "
                  f"embeddings; the text pathway learns nothing. Run src.precompute_text or set PROTGEN_TEXT_ENCODER.",
                  flush=True)
        embedder = DummyTextEmbedder(mcfg.text_dim, dcfg.max_text_tokens) if use_dummy \
            else HFTextEmbedder(TEXT_ENCODER, dcfg.max_text_tokens, device="cpu")

    token_budget = 2048 if args.smoke else ocfg.global_batch_tokens   # B per bucket = token_budget // bucket_len
    sampler = BucketedLengthSampler(ds.lengths, ds.n_sp_target, dcfg.length_buckets, token_budget,
                                    dcfg.p_swissprot, rank=env.rank, world=env.world_size, seed=ocfg.seed)
    # A live HF encoder holds a torch model that can't be forked to workers; the cache reader is fork-safe.
    nw = 0 if (args.smoke or isinstance(embedder, HFTextEmbedder)) else dcfg.num_workers
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=nw,
        collate_fn=make_collate(embedder, mcfg, dcfg.length_buckets, dcfg.max_text_tokens),
        **({"prefetch_factor": dcfg.prefetch_factor} if nw > 0 else {}))
    if env.is_main:
        print(f"[train] swissprot={len(sp_rows)} unannotated={len(unannot)} mixed={len(ds)} "
              f"text={'cache' if has_cache else type(embedder).__name__} tok_budget={token_budget} "
              f"batches/epoch/rank={len(sampler)}\n[train] buckets: {sampler.summary()}", flush=True)

    # --- optimizer + ipex; build the LR scheduler AFTER ipex.optimize so it binds to the optimizer
    #     that actually steps (removes the "optimizer.step() overridden after scheduler init" warning
    #     and any doubt about whether the schedule reaches the stepping optimizer) ---
    opt = torch.optim.AdamW(model.parameters(), lr=ocfg.lr, weight_decay=ocfg.weight_decay, betas=(0.9, 0.98))
    use_ipex = ocfg.use_ipex and not args.no_ipex
    applied_ipex = ipex is not None and dev.type == "xpu" and use_ipex
    if applied_ipex:
        model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16)
    if env.is_main:
        print(f"[train] ipex.optimize {'ON' if applied_ipex else 'OFF'} (device={dev.type}, "
              f"requested={use_ipex}, ipex={'available' if ipex is not None else 'missing'})", flush=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, ocfg.warmup_steps, ocfg.total_steps))

    # --- resume (conv-free model -> ipex keeps state_dict keys stable; opt-state load best-effort) ---
    # Rank 0 alone resolves latest.txt and reads the file; the bytes go out over the fabric. Every
    # rank then deserialises an identical payload, so no post-load broadcast_parameters is needed.
    # weights_only=True is passed explicitly: it is already the default on this torch (2.10) and the
    # payload is safe under it -- tensors, AdamW state, and LambdaLR state (whose lr_lambdas
    # serialise as None for plain functions).
    start_step = 0
    ckpt_path = find_latest_ckpt(CKPT_DIR) if (env.is_main and not args.fresh) else None
    raw = None if args.fresh else broadcast_checkpoint_bytes(ckpt_path, dev)
    if raw is not None:
        ck = torch.load(io.BytesIO(raw), map_location=dev, weights_only=True)
        del raw
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as ex:
            if env.is_main:
                print(f"[ckpt] optimizer state not restored ({ex}); re-warming moments.", flush=True)
        sched.load_state_dict(ck["sched"])
        start_step = int(ck["step"]) + 1
        if env.is_main:
            print(f"[ckpt] resumed from {ckpt_path}: continuing at step {start_step}", flush=True)

    n_grad = preallocate_grad_buffer(model, dev)
    if env.is_main and n_grad:
        print(f"[train] all-reduce buffer preallocated: {4 * (n_grad + 1) / 1e6:.0f}MB fp32", flush=True)

    total = args.max_steps or (60 if args.smoke else ocfg.total_steps)
    log_every = 10 if args.smoke else ocfg.log_every
    use_amp = dev.type in ("xpu", "cuda")
    step = start_step
    tok_win, steps_win, t_win = 0, 0, time.perf_counter()      # throughput window (reset each log)
    skipped, skip_streak = 0, 0                                # non-finite-gradient updates dropped
    model.train()

    while step < total:
        sampler.set_epoch(step)
        for batch in loader:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            n_tok = batch["tokens"].numel()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, m = training_step(model, batch, p_uncond=ocfg.p_uncond,
                                        beta=ocfg.beta, sub_probs=sub_probs, beta_schedule=ocfg.beta_schedule,
                                        pad_loss_weight=ocfg.pad_loss_weight)
            loss.backward()
            nonfinite = average_gradients(model)
            if nonfinite:
                skipped += 1
                skip_streak += 1
                if env.is_main:
                    print(f"[warn] step {step}: non-finite gradient -> update skipped "
                          f"({skip_streak} consecutive, {skipped} total)", flush=True)
                if skip_streak >= ocfg.max_skip_streak:
                    raise RuntimeError(
                        f"{skip_streak} consecutive non-finite gradients ending at step {step}; "
                        f"aborting rather than spinning. Last good checkpoint is in {CKPT_DIR}.")
            else:
                skip_streak = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg.grad_clip)
                opt.step()
            sched.step()
            tok_win += n_tok
            steps_win += 1
            if env.is_main and step % log_every == 0:
                _device_sync(dev)                              # finish queued work before reading the clock
                dt = max(time.perf_counter() - t_win, 1e-9)
                rate = f"{tok_win / dt / 1e3:.0f}k tok/s | {dt / steps_win:.2f}s/step"
                peak = _peak_mem_gb(dev)
                mem = f" | peak {peak:.1f}GB" if peak else ""
                # training_step returns on-device 0-d tensors; they are pulled to the host ONLY here.
                skip = f" | skipped {skipped}" if skipped else ""
                print(f"step {step:>7} | loss {float(m['oadm']):.3f} "
                      f"| cond {int(m['n_cond'])}/{int(m['n_labelled'])} "
                      f"| lr {sched.get_last_lr()[0]:.2e} | {rate}{mem}{skip}", flush=True)
                tok_win, steps_win, t_win = 0, 0, time.perf_counter()
            if step > 0 and step % ocfg.ckpt_every == 0:
                save_checkpoint(model, opt, sched, step, CKPT_DIR, env)
            step += 1
            if step >= total:
                break
    save_checkpoint(model, opt, sched, step - 1, CKPT_DIR, env)     # final checkpoint
    if env.is_main:
        print(f"[train] done at step {step}", flush=True)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
