"""Distributed bootstrap for Aurora (Intel XPU + oneCCL).

Mirrors the conventions of mini-embed-filip/src/dist.py: one rank per GPU *tile*
(Aurora = 6 Max-1550 x 2 tiles = 12 tiles/node), topology read from MPI first then
PALS/PMI env, `xccl` backend preferred (falls back to `ccl`). Degrades to a single-rank
world on a laptop so the same code smoke-tests on CPU.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import torch

try:
    import intel_extension_for_pytorch as ipex  # noqa: F401  (registers the xpu backend)
except Exception:
    ipex = None
try:
    import oneccl_bindings_for_pytorch  # noqa: F401  (registers the "ccl" PG backend)
except Exception:
    pass


@dataclass
class DistEnv:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def _first_env(*names, default=0):
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return int(v)
    return default


def _detect_topology():
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world = comm.Get_size()
        if world > 1:
            return comm.Get_rank(), world, comm.Split_type(MPI.COMM_TYPE_SHARED).Get_rank()
    except Exception:
        pass
    world = _first_env("PALS_NRANKS", "PMI_SIZE", "WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", default=1)
    rank = _first_env("PALS_RANKID", "PMI_RANK", "RANK", "OMPI_COMM_WORLD_RANK", default=0)
    local = _first_env("PALS_LOCAL_RANKID", "MPI_LOCALRANKID", "LOCAL_RANK",
                       "OMPI_COMM_WORLD_LOCAL_RANK", default=0)
    return rank, world, local


def _pick_device(local_rank: int, name: str) -> torch.device:
    if name not in ("auto", "xpu") and not name.startswith("xpu"):
        return torch.device(name)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        idx = local_rank % max(torch.xpu.device_count(), 1)
        torch.xpu.set_device(idx)
        return torch.device(f"xpu:{idx}")
    if torch.cuda.is_available():
        idx = local_rank % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(idx)
        return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def _xpu_backend() -> str:
    try:
        if getattr(torch.distributed, "is_xccl_available", lambda: False)():
            return "xccl"
    except Exception:
        pass
    try:
        import oneccl_bindings_for_pytorch  # noqa: F401
        return "ccl"
    except Exception:
        return "xccl"


def init_distributed(device_name: str = "auto") -> DistEnv:
    rank, world, local = _detect_topology()
    device = _pick_device(local, device_name)

    # Aurora footguns, both set by scripts/pbs_common.sh. An interactive session that did not source
    # it runs a DIFFERENT execution model, which silently invalidates any attempt to reproduce a job.
    if device.type == "xpu" and rank == 0:
        # frameworks defaults ONEAPI_DEVICE_SELECTOR to opencl+level_zero, doubling the device list
        # so ranks mis-pin and the GPU aborts.
        if "opencl" in os.environ.get("ONEAPI_DEVICE_SELECTOR", "").lower():
            print("[dist] WARNING: ONEAPI_DEVICE_SELECTOR exposes OpenCL+Level-Zero; set level_zero:gpu.",
                  flush=True)
        # Without FLAT, one "device" is a whole 2-tile GPU with implicit scaling across both tiles --
        # half the device count, mis-pinned ranks, and work spread over two tiles instead of one.
        hier = os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "")
        if hier.upper() != "FLAT":
            print(f"[dist] WARNING: ZE_FLAT_DEVICE_HIERARCHY={hier or '<unset>'}, want FLAT. Each device "
                  f"is then a whole 2-tile GPU with implicit scaling, not one tile -- a repro run like "
                  f"this does NOT match a FLAT training job.", flush=True)
        print(f"[dist] xpu: {torch.xpu.device_count()} device(s) visible "
              f"(expect 12 tiles/node on Aurora under FLAT), using {device}", flush=True)

    if world <= 1:
        return DistEnv(0, 1, 0, device, "none")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"], os.environ["WORLD_SIZE"], os.environ["LOCAL_RANK"] = str(rank), str(world), str(local)
    backend = _xpu_backend() if device.type == "xpu" else ("nccl" if device.type == "cuda" else "gloo")
    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(backend=backend, init_method="env://",
                                                 world_size=world, rank=rank,
                                                 device_id=device if device.type in ("xpu", "cuda") else None)
        except TypeError:
            torch.distributed.init_process_group(backend=backend, init_method="env://",
                                                 world_size=world, rank=rank)
    if rank == 0:
        print(f"[dist] backend={backend} world={world} rank0 device={device}", flush=True)
    return DistEnv(rank, world, local, device, backend)


def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def cleanup():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def broadcast_parameters(model, src: int = 0):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for p in model.parameters():
            torch.distributed.broadcast(p.data, src=src)


def broadcast_checkpoint_bytes(path, device, src: int = 0):
    """Rank `src` reads `path` and broadcasts the raw bytes; every rank returns identical bytes
    (or None if `path` is None on `src`, i.e. there is no checkpoint to resume).

    Only ONE rank touches the filesystem. Having all 192 ranks torch.load the same ~650MB checkpoint
    off flare at once is an I/O storm that can stall a job for minutes; a fabric broadcast is far
    cheaper. It also makes the resume decision unanimous by construction -- if each rank resolved
    latest.txt itself, a rank that disagreed about whether a checkpoint exists would hang the job.
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        if path is None:
            return None
        with open(path, "rb") as f:
            return f.read()

    import numpy as np
    rank = torch.distributed.get_rank()
    n = torch.zeros(1, dtype=torch.int64, device=device)
    arr = None
    if rank == src:
        if path is not None:
            arr = np.fromfile(path, dtype=np.uint8)
            n[0] = arr.size
        else:
            n[0] = -1
    torch.distributed.broadcast(n, src=src)
    size = int(n.item())
    if size < 0:
        return None
    buf = torch.empty(size, dtype=torch.uint8, device=device)
    if rank == src:
        buf.copy_(torch.from_numpy(arr))
    torch.distributed.broadcast(buf, src=src)
    return buf.to("cpu").numpy().tobytes()


_GRAD_BUF = None        # persistent [n_grad_elems + 1] fp32 buffer: coalesced grads, then a flag


def preallocate_grad_buffer(model, device) -> int:
    """Allocate the all-reduce buffer BEFORE training touches the allocator; returns its elements.

    Without this the buffer is created on the first average_gradients call, i.e. after a full forward
    and backward have already churned the heap, so its address is whatever happens to be free then.
    oneCCL caches L0 registrations and IPC handles keyed by pointer, so the one buffer it touches
    every step is exactly the thing whose address should be fixed for the life of the run and taken
    from a clean heap. Cheap insurance: ~220MB that was going to be resident anyway.

    Sized from every parameter. average_gradients re-derives the size from the grads that actually
    exist and reallocates if they disagree, so a mismatch degrades to the old behaviour rather than
    corrupting anything.
    """
    global _GRAD_BUF
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return 0
    n = sum(p.numel() for p in model.parameters())
    _GRAD_BUF = torch.empty(n + 1, dtype=torch.float32, device=device)
    return n


def average_gradients(model) -> bool:
    """All-reduce the gradients. Returns True if ANY rank produced a non-finite gradient this step.

    Safe here because the param-grad set is identical across ranks every step: the conditioning
    pathway always fires via the learned null token, so no param is ever conditionally unused.

    ONE COLLECTIVE PER STEP, and that is deliberate. Every grad goes into a single fp32 buffer
    regardless of its own dtype, and the non-finite flag rides in the buffer's last element rather
    than taking an all-reduce of its own. At 192 ranks over 16 nodes each additional collective is
    another size-dependent algorithm path inside oneCCL, and small-message paths differ from
    large-message ones; a single large fp32 all-reduce is the shape this job is known to sustain.
    Carrying the flag in-band also makes the skip decision unanimous for free -- every rank reads
    the same reduced element -- with no second collective to keep in sync.

    MIXED GRAD DTYPES ARE NORMAL: ipex.optimize(dtype=bfloat16) casts Linear weights to bf16 but
    deliberately leaves norms, gates and other small parameters in fp32. copy_ upcasts on the way
    in and downcasts on the way out. Reducing in fp32 matters on its own -- accumulating 192 bf16
    values loses ~8% of the value to the 8-bit mantissa.

    The buffer is allocated ONCE and reused. Building it per step (the original
    _flatten_dense_tensors version) freed a ~220MB block back to the caching allocator the instant
    the collective was enqueued; oneCCL runs on its own queue, so the allocator could hand that
    block to the next compute kernel while CCL was still reading it -- a use-after-free that
    surfaces as an intermittent, rank-local GPU page fault.
    """
    global _GRAD_BUF
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    world = torch.distributed.get_world_size()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return False
    device = grads[0].device
    if any(g.device != device for g in grads):
        raise RuntimeError("average_gradients needs all grads on one device; got "
                           f"{sorted({str(g.device) for g in grads})}")

    n = sum(g.numel() for g in grads)
    if _GRAD_BUF is None or _GRAD_BUF.numel() != n + 1 or _GRAD_BUF.device != device:
        _GRAD_BUF = torch.empty(n + 1, dtype=torch.float32, device=device)
    flat = _GRAD_BUF[:n]

    off = 0
    for g in grads:
        flat[off:off + g.numel()].copy_(g.reshape(-1))         # upcasts bf16 -> fp32
        off += g.numel()
    _GRAD_BUF[n] = (~torch.isfinite(flat).all()).float()       # in-band flag, stays on device

    torch.distributed.all_reduce(_GRAD_BUF, op=torch.distributed.ReduceOp.SUM)

    if bool(_GRAD_BUF[n].item() > 0):                          # the one host sync
        return True                                            # caller skips; don't write back garbage
    flat /= world
    off = 0
    for g in grads:
        g.copy_(flat[off:off + g.numel()].view_as(g))          # downcasts back to the grad dtype
        off += g.numel()
    return False
