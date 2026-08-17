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
import sys
import math
import time
import argparse
import torch

from config import CFG, SWISSPROT_CSV, UNANNOTATED_SHARDS, TEXT_ENCODER, TEXT_CACHE, BLOSUM_MAT, CKPT_DIR
from .dist import (init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
                   broadcast_checkpoint_bytes, preallocate_grad_buffer)
from .recurrent_oadm import RecurrentOADM, count_params
from .objective import training_step
from .sampler import generate
from .blosum import blosum_sub_probs
from .data import (ProteinTokenizer, DummyTextEmbedder, HFTextEmbedder, CacheTextEmbedder, ProteinShards,
                   load_swissprot, MixedProteinDataset, BucketedLengthSampler, make_collate)

try:
    import intel_extension_for_pytorch as ipex
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


# --- periodic generative eval -------------------------------------------------------------
def _rng_snapshot(dev):
    return torch.random.get_rng_state(), (torch.xpu.get_rng_state() if dev.type == "xpu" else
                                          torch.cuda.get_rng_state() if dev.type == "cuda" else None)


def _rng_restore(snap, dev):
    cpu_state, dev_state = snap
    torch.random.set_rng_state(cpu_state)
    if dev_state is not None:
        (torch.xpu if dev.type == "xpu" else torch.cuda).set_rng_state(dev_state)


@torch.no_grad()
def eval_sample_lengths(model, dev, n, canvas, steps, seed=1234):
    """Sample `n` sequences unconditionally and report their length distribution.

    Length is emergent here (the model decides by placing EOS), so it is the first generative
    property worth watching and the loss curve cannot tell you about it. Returns
    (mean, stdev, n_no_eos) where n_no_eos counts rows that never emitted EOS -- those come back as
    the full canvas width, so without that count a "mean 512" is indistinguishable from a genuine
    long-sequence distribution. Hook oracle metrics onto the returned tokens later.

    RNG is snapshotted, fixed, and restored, which buys two things: the training stream is bit-identical
    whether or not eval runs (so eval settings never perturb the trajectory), and every eval draws the
    SAME decode noise, making the trend across checkpoints a paired comparison instead of a noisy one.

    Every rank runs this and only rank 0 prints. Ranks would otherwise idle inside the next collective
    while rank 0 sampled; doing it in lockstep costs the same wall time and avoids a 192-way straggler.
    (Giving each rank its own seed and all-reducing the moments would widen the sample to n*world for
    free, if tighter statistics are ever wanted.)
    """
    snap = _rng_snapshot(dev)
    torch.manual_seed(seed)
    try:
        _, lengths = generate(model, Lmax=canvas, batch_size=n, text_emb=None, cfg_weight=0.0,
                              n_steps=steps, temperature=1.0, gumbel_temp=0.1, greedy=False,
                              device=str(dev))
    finally:
        _rng_restore(snap, dev)
    lens = torch.tensor(lengths, dtype=torch.float32)
    return lens.mean().item(), lens.std().item(), int((lens >= canvas).sum())


def dump_mem_segments(dev, rank, step):
    """Print the caching allocator's segment address ranges for this rank.

    Why this matters: the GPU fault address has been IDENTICAL across every run so far -- three
    values whose low 44 bits all read 0xfffff0e00000, differing only in the byte that encodes the
    tile. That is one specific allocation, not a race on random memory. So the question collapses to
    a single bit: does that address fall INSIDE a PyTorch segment?

      inside  -> the memory is PyTorch's, and something outside PyTorch (oneCCL, through a cached
                 IPC handle) is reading it after the allocator recycled or released it. Fix belongs
                 in how we hand buffers to the collective.
      outside -> the memory is oneCCL's own (its device cache is 800MB, its scaleout buffer 1GB),
                 and our allocator churn is only the trigger. Fix belongs in the CCL configuration.

    Grep the job log for the fault address against these ranges.
    """
    snap = getattr(torch.xpu, "memory_snapshot", None) if dev.type == "xpu" else None
    if snap is None:
        print(f"[mem] rank {rank}: no memory_snapshot on this device type", file=sys.stderr, flush=True)
        return
    for s in snap():
        addr, size = s.get("address"), s.get("total_size", 0)
        if addr is not None:
            print(f"[mem] rank {rank} step {step} seg 0x{addr:012x}-0x{addr + size:012x} "
                  f"size {size} {s.get('segment_type', '?')}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run: few CSV rows, dummy text, a few steps")
    ap.add_argument("--fresh", action="store_true", help="ignore any existing checkpoint and start from step 0")
    ap.add_argument("--max-steps", type=int, default=None, help="override total_steps (e.g. debug-queue validation)")
    ap.add_argument("--no-ipex", action="store_true", help="skip ipex.optimize (eager XPU; robust to dynamic shapes)")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="override opt.eval_every for the sampling eval (0 disables it)")
    ap.add_argument("--grad-checkpoint", default=None, action=argparse.BooleanOptionalAction,
                    help="recurrence-loop gradient checkpointing (default off: see Config.grad_checkpoint "
                         "-- it currently page-faults the GPU inside backward at 16 nodes)")
    ap.add_argument("--drain-collectives", action="store_true",
                    help="diagnostic: barrier() after every gradient all-reduce, forcing oneCCL to "
                         "complete before the next step allocates. NB torch.xpu.synchronize() does "
                         "NOT do this -- it drains PyTorch's queues, not oneCCL's own. Use this to "
                         "test whether an in-flight collective racing the allocator is the fault.")
    ap.add_argument("--mem-snapshot", type=int, default=0, metavar="STEP",
                    help="diagnostic: at this step, every rank prints its allocator segment address "
                         "ranges. Compare against the GPU fault address to learn whether the faulting "
                         "memory is PyTorch's or oneCCL's -- see dump_mem_segments(). Use the step "
                         "just before the known fault (e.g. 19002).")
    ap.add_argument("--sync-stages", type=int, default=0, metavar="N",
                    help="diagnostic: device-sync and print the stage after each phase of the first N "
                         "steps, from EVERY rank. A GPU page fault aborts the process with no Python "
                         "traceback, so the last [stage] line a rank printed names the faulting phase. "
                         "Grep the .o log for the rank in the abort message. Costs throughput; the "
                         "syncs also serialise the step, so do not read tok/s from such a run.")
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
        # Report what actually happened. The old message read the config flag, so it printed "ON"
        # even on a CPU run or when the ipex import had failed -- exactly when you need to know.
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

    # Fix the all-reduce buffer's address before the first forward churns the heap (see dist.py).
    n_grad = preallocate_grad_buffer(model, dev)
    if env.is_main and n_grad:
        print(f"[train] all-reduce buffer preallocated: {4 * (n_grad + 1) / 1e6:.0f}MB fp32", flush=True)

    total = args.max_steps or (60 if args.smoke else ocfg.total_steps)
    log_every = 10 if args.smoke else ocfg.log_every
    # Smoke runs on CPU, so shrink the eval to something that still exercises the whole path.
    ev_every = args.eval_every if args.eval_every is not None else (20 if args.smoke else ocfg.eval_every)
    ev_n, ev_canvas, ev_steps = (8, 128, 8) if args.smoke else (ocfg.eval_n, ocfg.eval_canvas, ocfg.eval_steps)
    use_amp = dev.type in ("xpu", "cuda")
    step = start_step
    tok_win, steps_win, t_win = 0, 0, time.perf_counter()      # throughput window (reset each log)
    skipped, skip_streak = 0, 0                                # non-finite-gradient updates dropped
    t_start, t_eval_total = time.perf_counter(), 0.0           # for the measured eval-overhead figure
    model.train()

    def stage(name, shape):
        """Diagnostic breadcrumb: see --sync-stages."""
        if step - start_step < args.sync_stages:
            _device_sync(dev)
            print(f"[stage] rank {env.rank} step {step} {name} shape={shape}", file=sys.stderr, flush=True)
    while step < total:
        sampler.set_epoch(step)
        for batch in loader:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            shape = tuple(batch["tokens"].shape)
            n_tok = batch["tokens"].numel()                    # canvas tokens processed this step (B*L)
            stage("h2d", shape)
            if args.mem_snapshot and step == args.mem_snapshot:
                dump_mem_segments(dev, env.rank, step)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, m = training_step(model, batch, p_uncond=ocfg.p_uncond,
                                        beta=ocfg.beta, sub_probs=sub_probs, beta_schedule=ocfg.beta_schedule,
                                        pad_loss_weight=ocfg.pad_loss_weight)
            stage("forward", shape)
            loss.backward()
            stage("backward", shape)
            # Average BEFORE clipping: clipping first clips each rank's local gradient to grad_clip and
            # then averages, which is not global-norm clipping and biases the step toward the ranks with
            # the smallest gradients. Order matters at 192 ranks. The reduction also reports whether any
            # rank produced a non-finite gradient, in-band, so no second collective is needed.
            nonfinite = average_gradients(model)
            if args.drain_collectives:
                barrier()
            stage("allreduce", shape)
            # A NaN/Inf gradient makes clip_grad_norm_'s coefficient NaN, so opt.step() would turn EVERY
            # parameter NaN -- the run then burns node-hours on garbage and writes poisoned checkpoints.
            # Skip the update instead; the next iteration's zero_grad discards the bad grads, and params
            # and AdamW moments are untouched. Every rank read the same reduced flag, so the decision is
            # unanimous and no rank can diverge into a hang.
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
            sched.step()                                       # always: LR stays a function of `step`
            stage("step", shape)
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
            if ev_every and step > 0 and step % ev_every == 0:
                t_ev = time.perf_counter()
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
                    mu, sd, n_no_eos = eval_sample_lengths(model, dev, ev_n, ev_canvas, ev_steps)
                _device_sync(dev)
                dt_ev = time.perf_counter() - t_ev
                t_eval_total += dt_ev
                t_win += dt_ev                                 # keep eval out of the tok/s window
                if env.is_main:
                    frac = 100 * t_eval_total / max(time.perf_counter() - t_start, 1e-9)
                    print(f"[eval] step {step} | len mean {mu:.1f} sd {sd:.1f} | no-EOS "
                          f"{n_no_eos}/{ev_n} | {dt_ev:.1f}s ({frac:.1f}% of wall so far)", flush=True)
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
