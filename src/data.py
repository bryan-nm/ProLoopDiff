"""Data pipeline: mixed SwissProt (labelled/text) + TrEMBL (unlabelled) -> training_step batches.

Design (Aurora conventions from mini-embed-filip):
  * Protein sequences are tokenised on the fly (they are the cheap generation target).
  * Text is the expensive part: BiomedBERT token embeddings are PRECOMPUTED to a cache and read
    here (see precompute_text.py). A live/dummy embedder is provided for small runs and smoke tests.
  * Length bucketing keeps the PAD tail short -- PAD is a MODELLED, attended token (EOS marks length),
    so we pad each batch to its own max length, not a global max.
  * Batch dict matches objective.training_step: tokens, labelled, text_emb, text_keep.
"""
from __future__ import annotations
import os
import csv
import random
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset

from .blosum import AA                                  # canonical 20-AA alphabet, id == index

# Map rare/ambiguous residues onto standard AAs; unknown chars -> sequence rejected.
_RARE = {"B": "D", "Z": "E", "J": "L", "U": "C", "O": "K", "X": "A"}
_AA_ID = {c: i for i, c in enumerate(AA)}

# 256-entry byte->id table (255 = unmappable). Lets encode() run as a single C-level lookup instead of
# a per-char Python loop -- the difference that matters when tokenising ~150M UniRef90 sequences.
_INVALID = 255
_ENCODE_TABLE = np.full(256, _INVALID, dtype=np.uint8)
for _c, _i in _AA_ID.items():
    _ENCODE_TABLE[ord(_c)] = _i
for _r, _std in _RARE.items():
    _ENCODE_TABLE[ord(_r)] = _AA_ID[_std]


class ProteinTokenizer:
    def __init__(self, cfg):
        self.eos, self.pad, self.mask = cfg.eos_token_id, cfg.pad_token_id, cfg.mask_token_id

    def encode(self, seq: str) -> Optional[List[int]]:
        """Amino-acid string -> [ids..., EOS]. Returns None if the sequence has any unmappable char."""
        b = seq.strip().upper().encode("ascii", "replace")   # non-ascii -> '?' (63) -> _INVALID below
        if not b:
            return None
        mapped = _ENCODE_TABLE[np.frombuffer(b, dtype=np.uint8)]
        if (mapped == _INVALID).any():
            return None
        return mapped.tolist() + [self.eos]


# ---------------------------------------------------------------------------
# Text embedders. Training calls encode_samples(samples) -> (B,Tmax,text_dim) float, (B,Tmax) bool keep
# (labelled rows carry real text; unlabelled rows are all-False and use the model's null token).
# Precompute calls encode_batch(list[str]) on the live encoders.
# ---------------------------------------------------------------------------
class _StringEmbedder:
    """Shares encode_samples for the live (string-keyed) encoders below."""
    def encode_samples(self, samples):
        caps = [s["caption"] if s["labelled"] else "" for s in samples]
        emb, keep = self.encode_batch(caps)
        lab = torch.tensor([s["labelled"] for s in samples])
        return emb, keep & lab[:, None]


class DummyTextEmbedder(_StringEmbedder):
    """Deterministic per-caption random embeddings for local smoke tests (no BERT download)."""
    def __init__(self, dim=768, max_tokens=128):
        self.dim, self.max_tokens = dim, max_tokens

    def encode_batch(self, captions):
        B = len(captions)
        lens = [min(max(len(c.split()), 1), self.max_tokens) for c in captions]
        T = max(lens)
        emb = torch.zeros(B, T, self.dim)
        keep = torch.zeros(B, T, dtype=torch.bool)
        import hashlib
        for i, (c, n) in enumerate(zip(captions, lens)):
            seed = int(hashlib.md5(c.encode()).hexdigest()[:8], 16)   # stable across processes (unlike hash())
            g = torch.Generator().manual_seed(seed)
            emb[i, :n] = torch.randn(n, self.dim, generator=g)
            keep[i, :n] = True
        return emb, keep


class HFTextEmbedder(_StringEmbedder):
    """Frozen BiomedBERT (PubMedBERT-full) token embeddings. Used by precompute and small live runs."""
    def __init__(self, model_path, max_tokens=128, device="cpu"):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_tokens, self.device = max_tokens, device

    @torch.no_grad()
    def encode_batch(self, captions):
        enc = self.tok(list(captions), padding=True, truncation=True,
                       max_length=self.max_tokens, return_tensors="pt").to(self.device)
        out = self.model(**enc).last_hidden_state          # (B, T, hidden)
        return out.cpu(), enc["attention_mask"].bool().cpu()


class CacheTextEmbedder:
    """Reads BiomedBERT token embeddings precomputed by precompute_text.py (keeps BERT out of the loop).

    Cache layout (see precompute_text.py): text_h.f16 [total_tokens, hidden] memmap, text_offsets.i64
    [N+1] token offsets, fingerprint.json. Samples carry `text_idx` = canonical SwissProt row index.
    """
    def __init__(self, cache_dir):
        import json
        import numpy as np
        with open(os.path.join(cache_dir, "fingerprint.json")) as f:
            self.fp = json.load(f)
        self.H = self.fp["hidden"]
        self.off = np.fromfile(os.path.join(cache_dir, "text_offsets.i64"), dtype="int64")
        self.h = np.memmap(os.path.join(cache_dir, "text_h.f16"), dtype="float16", mode="r").reshape(-1, self.H)

    def __len__(self):
        return len(self.off) - 1

    def encode_samples(self, samples):
        rows, lens = [], []
        for s in samples:
            if s["labelled"]:
                a, b = int(self.off[s["text_idx"]]), int(self.off[s["text_idx"] + 1])
                rows.append(self.h[a:b])
                lens.append(b - a)
            else:
                rows.append(None)
                lens.append(0)
        T = max(lens) if any(lens) else 1
        B = len(samples)
        emb = torch.zeros(B, T, self.H)
        keep = torch.zeros(B, T, dtype=torch.bool)
        for i, (r, n) in enumerate(zip(rows, lens)):
            if n:
                emb[i, :n] = torch.from_numpy(r.astype("float32"))
                keep[i, :n] = True
        return emb, keep


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def load_swissprot(path, tokenizer, limit=None):
    """-> list of (accession, ids, caption), deterministic order (skips unmappable sequences).
    THIS is the canonical order: the text cache row index == position in this list. pair_ids.json
    records accessions so precompute and training can verify they agree."""
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cap_col = [c for c in r.fieldnames if "caption" in c][0]
        for row in r:
            ids = tokenizer.encode(row["protein_sequence"])
            if ids is not None:
                rows.append((row["primary_Accession"], ids, row[cap_col]))
            if limit and len(rows) >= limit:
                break
    return rows


class ProteinShards:
    """Reader for pre-tokenised unannotated-corpus shards (UniRef90): uint8 .bin of ids 0..19 + .idx
    of int64 offsets (see preprocess_fasta.py). Empty/absent -> len 0 (pure-SwissProt runs still work)."""
    def __init__(self, shard_dir, eos_id):
        import glob
        self.eos_id = eos_id
        self.bins = sorted(glob.glob(os.path.join(shard_dir, "*.bin"))) if shard_dir and os.path.isdir(shard_dir) else []
        self.offsets, self.data = [], []
        for b in self.bins:
            import numpy as np
            idx = np.fromfile(b[:-4] + ".idx", dtype="int64")
            self.data.append(np.memmap(b, dtype="uint8", mode="r"))
            self.offsets.append(idx)
        self.index = [(s, k) for s, off in enumerate(self.offsets) for k in range(len(off) - 1)]

    def __len__(self):
        return len(self.index)

    def get_len(self, i):                                   # O(1) from the offset index (no data read)
        s, k = self.index[i]
        return int(self.offsets[s][k + 1] - self.offsets[s][k]) + 1   # +1 for EOS

    def get(self, i):
        s, k = self.index[i]
        a, b = self.offsets[s][k], self.offsets[s][k + 1]
        return [int(x) for x in self.data[s][a:b]] + [self.eos_id]    # append EOS (shards store residues only)


class MixedProteinDataset(Dataset):
    """Unified index over labelled SwissProt (oversampled to hit p_swissprot) + the unlabelled corpus.
    SwissProt rows are (accession, ids, caption); the SwissProt list index doubles as the text-cache
    row index (text_idx), carried on each labelled sample."""
    def __init__(self, swissprot_rows, unannotated: Optional["ProteinShards"], p_swissprot: float):
        self.sp = swissprot_rows
        self.un = unannotated
        n_un = len(unannotated) if unannotated else 0
        if n_un == 0 or not self.sp:
            self.items = [("sp", i) for i in range(len(self.sp))] or [("un", i) for i in range(n_un)]
        else:
            n_sp_target = int(round(p_swissprot / (1 - p_swissprot) * n_un))
            self.items = [("sp", i % len(self.sp)) for i in range(n_sp_target)] + [("un", i) for i in range(n_un)]
            random.Random(0).shuffle(self.items)
        self.lengths = [len(self.sp[i][1]) if s == "sp" else self.un.get_len(i) for (s, i) in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        s, i = self.items[k]
        if s == "sp":
            _acc, ids, caption = self.sp[i]
            return {"ids": ids, "caption": caption, "labelled": True, "text_idx": i}
        return {"ids": self.un.get(i), "caption": None, "labelled": False, "text_idx": -1}


# ---------------------------------------------------------------------------
# Length bucketing + collate
# ---------------------------------------------------------------------------
class BucketedBatchSampler(torch.utils.data.Sampler):
    """Yield batches of similar-length indices, sharded across ranks. drop_last so per-rank step count matches."""
    def __init__(self, lengths, micro_batch, rank=0, world=1, shuffle=True, seed=0):
        self.lengths, self.mb, self.rank, self.world = lengths, micro_batch, rank, world
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [order[i:i + self.mb] for i in range(0, len(order), self.mb) if len(order[i:i + self.mb]) == self.mb]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        for b in batches[self.rank::self.world]:            # each rank takes every world-th batch
            yield b

    def __len__(self):
        return (len(self.lengths) // self.mb) // self.world


def make_collate(embedder, cfg_model):
    pad = cfg_model.pad_token_id

    def collate(samples):
        L = max(len(s["ids"]) for s in samples)
        B = len(samples)
        tokens = torch.full((B, L), pad, dtype=torch.long)
        for i, s in enumerate(samples):
            tokens[i, :len(s["ids"])] = torch.tensor(s["ids"])
        labelled = torch.tensor([s["labelled"] for s in samples])
        # Text: cache-gather by text_idx (precompute) or live encode; unlabelled rows -> zero + keep False.
        text_emb, text_keep = embedder.encode_samples(samples)
        return {"tokens": tokens, "labelled": labelled, "text_emb": text_emb, "text_keep": text_keep}

    return collate
