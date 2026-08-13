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
    of int64 offsets (see preprocess_fasta.py). Empty/absent -> len 0 (pure-SwissProt runs still work).

    Scale note: at ~100M+ sequences we must NOT build a per-sequence Python index. We keep only the
    per-shard offset arrays + a cumulative count, and locate item i by searchsorted (O(log #shards))."""
    def __init__(self, shard_dir, eos_id):
        import glob
        self.eos_id = eos_id
        bins = sorted(glob.glob(os.path.join(shard_dir, "*.bin"))) if shard_dir and os.path.isdir(shard_dir) else []
        self.offsets, self.data = [], []
        for b in bins:
            self.offsets.append(np.fromfile(b[:-4] + ".idx", dtype="int64"))
            self.data.append(np.memmap(b, dtype="uint8", mode="r"))
        counts = [len(off) - 1 for off in self.offsets]
        self.cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64) if counts else np.array([0], np.int64)
        self._n = int(self.cum[-1])

    def __len__(self):
        return self._n

    def _locate(self, i):
        s = int(np.searchsorted(self.cum, i, side="right") - 1)
        return s, i - int(self.cum[s])

    def get_len(self, i):
        s, k = self._locate(i)
        return int(self.offsets[s][k + 1] - self.offsets[s][k]) + 1   # +1 for EOS

    def get(self, i):
        s, k = self._locate(i)
        a, b = int(self.offsets[s][k]), int(self.offsets[s][k + 1])
        return self.data[s][a:b].tolist() + [self.eos_id]            # append EOS (shards store residues only)

    def lengths_array(self) -> np.ndarray:
        """All sequence lengths (+EOS) as one int32 array, vectorised from the offset arrays."""
        if not self.offsets:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate([np.diff(off).astype(np.int32) + 1 for off in self.offsets])


class MixedProteinDataset(Dataset):
    """Unified index over labelled SwissProt (oversampled to hit p_swissprot) + the unlabelled corpus.
    No per-item Python list: sources are resolved arithmetically, and lengths are one numpy array.
    SwissProt list index doubles as the text-cache row index (text_idx), carried on each labelled sample."""
    def __init__(self, swissprot_rows, unannotated: Optional["ProteinShards"], p_swissprot: float):
        self.sp = swissprot_rows
        self.un = unannotated
        self._nsp = len(self.sp)
        n_un = len(unannotated) if unannotated else 0
        self.n_un = n_un
        if n_un == 0 or self._nsp == 0:
            self.n_sp_target = self._nsp
        else:
            self.n_sp_target = int(round(p_swissprot / (1 - p_swissprot) * n_un))  # oversample SwissProt

        sp_len = np.array([len(r[1]) for r in self.sp], dtype=np.int32) if self._nsp else np.zeros(0, np.int32)
        sp_part = sp_len[np.arange(self.n_sp_target) % self._nsp] if (self.n_sp_target and self._nsp) \
            else np.zeros(0, np.int32)
        un_part = unannotated.lengths_array() if (unannotated and n_un) else np.zeros(0, np.int32)
        self.lengths = np.concatenate([sp_part, un_part])      # (n_sp_target + n_un,) int32

    def __len__(self):
        return int(self.n_sp_target + self.n_un)

    def __getitem__(self, k):
        if k < self.n_sp_target:                               # SwissProt block (oversampled, wraps)
            i = k % self._nsp
            _acc, ids, caption = self.sp[i]
            return {"ids": ids, "caption": caption, "labelled": True, "text_idx": i}
        j = k - self.n_sp_target                               # unannotated block
        return {"ids": self.un.get(j), "caption": None, "labelled": False, "text_idx": -1}


# ---------------------------------------------------------------------------
# Length bucketing + collate
# ---------------------------------------------------------------------------
class BucketedBatchSampler(torch.utils.data.Sampler):
    """Yield batches of similar-length indices, sharded across ranks. drop_last so per-rank step count
    matches. The global argsort is computed ONCE (numpy); each epoch only permutes batch order."""
    def __init__(self, lengths, micro_batch, rank=0, world=1, shuffle=True, seed=0):
        self.mb, self.rank, self.world = micro_batch, rank, world
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        lengths = np.asarray(lengths)
        # Sort by length, breaking ties RANDOMLY (seeded identically on every rank). A stable sort would
        # keep the concatenated SwissProt block ahead of the corpus block within each length, making
        # batches all-labelled or all-unlabelled; the random tiebreak interleaves the two sources so
        # each batch carries ~p_swissprot labelled rows. Same-length only, so padding stays tight.
        tiebreak = np.random.default_rng(seed).integers(0, 1 << 20, len(lengths), dtype=np.int64)
        self.order = np.lexsort((tiebreak, lengths)).astype(np.int32)   # primary key = lengths, indices < 2^31
        self.n_batches = len(self.order) // self.mb

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        perm = (np.random.default_rng(self.seed + self.epoch).permutation(self.n_batches)
                if self.shuffle else np.arange(self.n_batches))
        for b in perm[self.rank::self.world]:               # each rank takes every world-th batch
            s = int(b) * self.mb
            yield self.order[s:s + self.mb].tolist()

    def __len__(self):
        return self.n_batches // self.world


class BucketedLengthSampler(torch.utils.data.Sampler):
    """Length-bucketed, fixed-composition, token-budgeted batches. Each batch draws from ONE length
    bucket and has a fixed (B, n_lab, n_unlab) for that bucket, arranged [labelled..., unlabelled...].
    -> ONE static shape per bucket (a handful total) so XPU compiles each once instead of per length.

    Per-bucket batch size B = token_budget // bucket_len, so short buckets pack more sequences at ~constant
    memory (and the long bucket stays memory-safe). Labelled dataset indices are [0, n_sp_target);
    unlabelled are [n_sp_target, n_total). `lengths` is the dataset's per-item length array."""
    def __init__(self, lengths, n_sp_target, buckets, token_budget, p_swissprot,
                 rank=0, world=1, seed=0, filip_min=2):
        self.rank, self.world, self.seed, self.epoch = rank, world, seed, 0
        buckets = list(buckets)
        lengths = np.asarray(lengths)
        n_total = len(lengths)
        is_lab = np.arange(n_total) < n_sp_target
        binid = np.clip(np.searchsorted(buckets, lengths, side="left"), 0, len(buckets) - 1)
        self.plan = []
        for bi, L in enumerate(buckets):
            in_b = binid == bi
            lab_pool = np.where(in_b & is_lab)[0].astype(np.int32)
            un_pool = np.where(in_b & ~is_lab)[0].astype(np.int32)
            if len(lab_pool) == 0 and len(un_pool) == 0:
                continue
            B = max(1, token_budget // L)
            if len(lab_pool) == 0:
                n_lab, n_unlab = 0, B
            elif len(un_pool) == 0:
                n_lab, n_unlab = B, 0
            else:
                n_lab = min(len(lab_pool), max(filip_min, round(B * p_swissprot)))
                n_lab = min(n_lab, B - 1)                          # leave >=1 unlabelled slot
                n_unlab = B - n_lab
            nb_lab = (len(lab_pool) // n_lab) if n_lab else (1 << 60)
            nb_un = (len(un_pool) // n_unlab) if n_unlab else (1 << 60)
            n_batches = int(min(nb_lab, nb_un))
            if n_batches:
                self.plan.append(dict(L=L, B=B, n_lab=n_lab, n_unlab=n_unlab,
                                      lab_pool=lab_pool, un_pool=un_pool, n_batches=n_batches))
        self.total = sum(p["n_batches"] for p in self.plan)

    def set_epoch(self, e):
        self.epoch = e

    def summary(self):
        return "  ".join(f"L{p['L']}:B{p['B']}({p['n_lab']}L+{p['n_unlab']}U)x{p['n_batches']}" for p in self.plan)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)          # same on every rank -> consistent batches
        perms = [(rng.permutation(len(p["lab_pool"])).astype(np.int32) if p["n_lab"] else None,
                  rng.permutation(len(p["un_pool"])).astype(np.int32) if p["n_unlab"] else None)
                 for p in self.plan]
        bucket_ids = np.concatenate([np.full(p["n_batches"], bi, np.int32) for bi, p in enumerate(self.plan)])
        batch_ids = np.concatenate([np.arange(p["n_batches"], dtype=np.int32) for p in self.plan])
        sh = rng.permutation(len(bucket_ids))                        # interleave buckets, shuffle order
        bucket_ids, batch_ids = bucket_ids[sh], batch_ids[sh]
        for k in range(self.rank, len(bucket_ids), self.world):
            bi, i = int(bucket_ids[k]), int(batch_ids[k])
            p = self.plan[bi]
            lp, up = perms[bi]
            batch = []
            if p["n_lab"]:
                batch += p["lab_pool"][lp[i * p["n_lab"]:(i + 1) * p["n_lab"]]].tolist()
            if p["n_unlab"]:
                batch += p["un_pool"][up[i * p["n_unlab"]:(i + 1) * p["n_unlab"]]].tolist()
            yield batch

    def __len__(self):
        return self.total // self.world


def make_collate(embedder, cfg_model, buckets, max_text_tokens):
    pad = cfg_model.pad_token_id
    buckets = list(buckets)

    def collate(samples):
        B = len(samples)
        maxlen = max(len(s["ids"]) for s in samples)
        L = next((b for b in buckets if b >= maxlen), buckets[-1])       # this batch's bucket length
        tokens = torch.full((B, L), pad, dtype=torch.long)
        for i, s in enumerate(samples):
            ids = s["ids"][:L]
            tokens[i, :len(ids)] = torch.tensor(ids)
        labelled = torch.tensor([s["labelled"] for s in samples])
        text_emb, text_keep = embedder.encode_samples(samples)           # (B, Tb, H), (B, Tb)
        Tb = text_emb.shape[1]                                           # pad/trim text to FIXED length
        if Tb < max_text_tokens:
            text_emb = torch.nn.functional.pad(text_emb, (0, 0, 0, max_text_tokens - Tb))
            text_keep = torch.nn.functional.pad(text_keep, (0, max_text_tokens - Tb))
        elif Tb > max_text_tokens:
            text_emb, text_keep = text_emb[:, :max_text_tokens], text_keep[:, :max_text_tokens]
        return {"tokens": tokens, "labelled": labelled, "text_emb": text_emb, "text_keep": text_keep}

    return collate
