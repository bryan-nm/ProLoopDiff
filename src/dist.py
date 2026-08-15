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

    # Aurora footgun: frameworks defaults ONEAPI_DEVICE_SELECTOR to opencl+level_zero, doubling the
    # device list so ranks mis-pin and the GPU aborts. The launch script must set level_zero:gpu.
    if device.type == "xpu" and rank == 0 and "opencl" in os.environ.get("ONEAPI_DEVICE_SELECTOR", "").lower():
        print("[dist] WARNING: ONEAPI_DEVICE_SELECTOR exposes OpenCL+Level-Zero; set level_zero:gpu.", flush=True)

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


_FLAG_BUF = None        # persistent 1-element buffer for any_rank


def any_rank(flag, device) -> bool:
    """OR a per-rank flag across the world and return it on the host.

    `flag` is a Python bool or any 0-d/1-element tensor (nonzero = True). Costs one tiny collective
    and exactly ONE device->host sync, so the caller can make a control-flow decision (skip a step,
    abort) that is guaranteed unanimous. Ranks that disagreed about whether to run a collective would
    deadlock until the walltime, which is why this is a guarantee and not an argument.
    """
    global _FLAG_BUF
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(flag) if isinstance(flag, bool) else bool(flag.reshape(-1)[0].item())
    if _FLAG_BUF is None or _FLAG_BUF.device != device:
        _FLAG_BUF = torch.zeros(1, dtype=torch.float32, device=device)
    if isinstance(flag, bool):
        _FLAG_BUF.fill_(1.0 if flag else 0.0)
    else:
        _FLAG_BUF.copy_(flag.reshape(-1)[:1].float())          # stays on device; no sync yet
    torch.distributed.all_reduce(_FLAG_BUF, op=torch.distributed.ReduceOp.MAX)
    return bool(_FLAG_BUF.item() > 0)                          # the one sync


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


_GRAD_BUF = None        # persistent coalescing buffer for average_gradients (see below)


def average_gradients(model):
    """Manual all-reduce of grads coalesced into one collective (mini-embed convention).
    Safe here because the param-grad set is identical across ranks every step: the conditioning
    pathway always fires via the learned null token, so no param is ever conditionally unused.

    The coalescing buffer is allocated ONCE and reused. The previous version built it per step with
    _flatten_dense_tensors and dropped it on return, so a ~100MB block was freed back to the caching
    allocator the instant the collective was enqueued. oneCCL runs the collective on its own queue,
    so the allocator could hand that block to the next compute kernel while CCL was still reading it
    -- a use-after-free that shows up as an intermittent, rank-local GPU page fault. Reusing one
    buffer removes both the hazard and ~100MB of alloc/free churn (and fragmentation) per step.
    """
    global _GRAD_BUF
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    world = torch.distributed.get_world_size()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return
    dtype, device = grads[0].dtype, grads[0].device
    if any(g.dtype != dtype or g.device != device for g in grads):
        raise RuntimeError("average_gradients needs one dtype/device for all grads; got "
                           f"{sorted({str(g.dtype) for g in grads})} on "
                           f"{sorted({str(g.device) for g in grads})}")
    n = sum(g.numel() for g in grads)
    if _GRAD_BUF is None or _GRAD_BUF.numel() != n or _GRAD_BUF.dtype != dtype or _GRAD_BUF.device != device:
        _GRAD_BUF = torch.empty(n, dtype=dtype, device=device)

    off = 0
    for g in grads:
        _GRAD_BUF[off:off + g.numel()].copy_(g.reshape(-1))
        off += g.numel()
    torch.distributed.all_reduce(_GRAD_BUF, op=torch.distributed.ReduceOp.SUM)
    _GRAD_BUF /= world
    off = 0
    for g in grads:
        g.copy_(_GRAD_BUF[off:off + g.numel()].view_as(g))
        off += g.numel()
