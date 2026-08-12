"""Shard-directory hygiene for the distributed precompute (trimmed from mini-embed-filip/src/shards.py).

Every rank writes `<name>.<rank>.<ext>` into a shard dir, then rank 0 concatenates in rank order.
Two defences against a re-run at a different world size leaving stale shards behind:
  reset_shard_dir()   PREVENTION: rank 0 empties the dir before any rank writes, then all barrier.
  check_shard_world() DETECTION: each shard meta stamps its world size; merge rejects a mixed set.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

from .dist import barrier


def reset_shard_dir(shards_dir, env) -> int:
    shards_dir = Path(shards_dir)
    removed = 0
    if env.is_main:
        shards_dir.mkdir(parents=True, exist_ok=True)
        for p in glob.glob(str(shards_dir / "*")):
            if os.path.isfile(p):
                os.remove(p)
                removed += 1
        if removed:
            print(f"[shards] cleared {removed} stale file(s) from {shards_dir}", flush=True)
    barrier()
    return removed


def check_shard_world(metas: list, shards_dir, expected=None) -> None:
    worlds = {m["world"] for m in metas if "world" in m}
    if len(worlds) > 1:
        raise RuntimeError(
            f"{shards_dir} holds shards from {len(worlds)} runs (world sizes {sorted(worlds)}). "
            f"A re-run at a smaller world size overwrites low-numbered shards and leaves the rest.\n"
            f"    rm -rf {shards_dir}\nand re-run the encode.")
    if worlds:
        (world,) = worlds
        if len(metas) != world:
            raise RuntimeError(f"{shards_dir}: {len(metas)} shards but world={world} (a rank died / polluted dir).\n"
                               f"    rm -rf {shards_dir}\nand re-run.")
        if expected is not None and world != expected:
            raise RuntimeError(f"{shards_dir}: shards from a {world}-rank run, but this is {expected} ranks.")
