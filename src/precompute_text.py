"""Precompute BiomedBERT token embeddings for the SwissProt captions -> a cache the trainer reads.

Keeps BERT out of the training loop (restores DataLoader num_workers > 0). Distributed like
mini-embed-filip/src/precompute.py: each rank encodes a CONTIGUOUS slice of the canonical SwissProt
order (so global row order is preserved), writes per-rank shards, then rank 0 merges in rank order.

Canonical order == data.load_swissprot(...); the text-cache row index == that list position, and the
trainer carries it as `text_idx`. pair_ids.json records accessions so alignment is verifiable.

Cache (written to config.TEXT_CACHE):
    text_h.f16       float16 [total_tokens, hidden]   (per-token BiomedBERT last_hidden_state)
    text_offsets.i64 int64   [N_rows + 1]             (row i tokens = h[off[i]:off[i+1]])
    pair_ids.json    [N_rows] accessions
    fingerprint.json {encoder, hidden, max_text_tokens, n_rows, format}

Run (Aurora): mpiexec ... python -m src.precompute_text --device xpu
Local smoke : PROTGEN_SWISSPROT_CSV=... python -m src.precompute_text --device cpu --limit 256
"""
from __future__ import annotations
import os
import json
import glob
import argparse
import numpy as np
import torch

from config import SWISSPROT_CSV, TEXT_ENCODER, TEXT_CACHE, CFG
from .dist import init_distributed, barrier
from .data import ProteinTokenizer, load_swissprot, HFTextEmbedder, DummyTextEmbedder
from .shards import reset_shard_dir, check_shard_world


def _shard_range(n, rank, world):
    per, rem = divmod(n, world)
    start = rank * per + min(rank, rem)
    return start, start + per + (1 if rank < rem else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (smoke)")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    env = init_distributed(args.device)
    mcfg, dcfg = CFG.model_config(), CFG.data
    shards_dir = os.path.join(TEXT_CACHE, "shards")

    tok = ProteinTokenizer(mcfg)
    rows = load_swissprot(SWISSPROT_CSV, tok, limit=args.limit)      # canonical order (acc, ids, caption)
    N = len(rows)
    if env.is_main:
        print(f"[precompute] {N} SwissProt rows | world={env.world_size} | cache={TEXT_CACHE}", flush=True)

    if not args.merge_only:
        reset_shard_dir(shards_dir, env)
        embedder = (DummyTextEmbedder(mcfg.text_dim, dcfg.max_text_tokens)
                    if not os.path.isdir(TEXT_ENCODER)
                    else HFTextEmbedder(TEXT_ENCODER, dcfg.max_text_tokens, device=str(env.device)))
        if env.is_main and isinstance(embedder, DummyTextEmbedder):
            print("[precompute] WARNING: TEXT_ENCODER not found -> DUMMY embeddings (wiring test only).", flush=True)

        s, e = _shard_range(N, env.rank, env.world_size)
        tag = f"{env.rank:05d}"
        h_parts, offs, accs = [], [0], []
        for b in range(s, e, args.batch_size):
            batch = rows[b:min(b + args.batch_size, e)]
            emb, keep = embedder.encode_batch([r[2] for r in batch])     # (B,T,H) fp, (B,T) keep
            for j, (row, k) in enumerate(zip(batch, keep)):
                tokens = emb[j][k].to(torch.float16).numpy()              # (Tj, H) real tokens only
                h_parts.append(tokens)
                offs.append(offs[-1] + tokens.shape[0])
                accs.append(row[0])
            if env.is_main and (b - s) % (args.batch_size * 20) == 0:
                print(f"[precompute][rank0] {b - s}/{e - s}", flush=True)
        np.concatenate(h_parts, axis=0).astype("float16").tofile(os.path.join(shards_dir, f"h.{tag}.f16")) \
            if h_parts else np.zeros((0, mcfg.text_dim), "float16").tofile(os.path.join(shards_dir, f"h.{tag}.f16"))
        np.asarray(offs, dtype="int64").tofile(os.path.join(shards_dir, f"off.{tag}.i64"))
        with open(os.path.join(shards_dir, f"meta.{tag}.json"), "w") as f:
            json.dump({"rank": env.rank, "world": env.world_size, "n": len(accs), "accs": accs}, f)
        print(f"[precompute][rank {env.rank}] wrote {len(accs)} rows", flush=True)
    barrier()

    # --- merge (rank 0) ---
    if env.is_main:
        metas = [json.load(open(m)) for m in sorted(glob.glob(os.path.join(shards_dir, "meta.*.json")))]
        check_shard_world(metas, shards_dir, expected=env.world_size if not args.merge_only else None)
        metas.sort(key=lambda m: m["rank"])
        H = mcfg.text_dim
        h_out = open(os.path.join(TEXT_CACHE, "text_h.f16"), "wb")
        global_off, pair_ids, base = [0], [], 0
        for m in metas:
            tag = f"{m['rank']:05d}"
            h = np.fromfile(os.path.join(shards_dir, f"h.{tag}.f16"), dtype="float16").reshape(-1, H)
            off = np.fromfile(os.path.join(shards_dir, f"off.{tag}.i64"), dtype="int64")
            h.tofile(h_out)
            for k in range(len(off) - 1):
                global_off.append(base + int(off[k + 1]))
            base += int(off[-1])
            pair_ids.extend(m["accs"])
        h_out.close()
        np.asarray(global_off, dtype="int64").tofile(os.path.join(TEXT_CACHE, "text_offsets.i64"))
        json.dump(pair_ids, open(os.path.join(TEXT_CACHE, "pair_ids.json"), "w"))
        json.dump({"encoder": TEXT_ENCODER, "hidden": H, "max_text_tokens": dcfg.max_text_tokens,
                   "n_rows": len(pair_ids), "format": "f16-per-token-v1"},
                  open(os.path.join(TEXT_CACHE, "fingerprint.json"), "w"))
        assert len(pair_ids) == N == len(global_off) - 1, (len(pair_ids), N, len(global_off) - 1)
        print(f"[precompute] merged {len(pair_ids)} rows -> {TEXT_CACHE} (total tokens {base})", flush=True)
    barrier()


if __name__ == "__main__":
    main()
