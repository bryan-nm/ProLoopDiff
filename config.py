"""Central configuration for the recurrent-OADM protein generator (Aurora conventions).

Mirrors mini-embed-filip: THIS FILE OWNS EVERY PATH. Job scripts must not pass model/dataset
locations on the command line. `python config.py` prints what a run resolves to; job scripts
banner that output so the .o log is the record.

PROTGEN_* env vars are the escape hatch for a workstation whose data lives elsewhere (what the
local smoke tests use). They are deliberately NOT set by any job script. PROTGEN_*_DIR swaps a
base dir; per-item vars override a single path.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.recurrent_oadm import Config as ModelConfig

REPO_ROOT = Path(__file__).resolve().parent

# --- base dirs (Aurora flare defaults; override for workstation smoke tests) ---
MODELS_DIR = os.environ.get("PROTGEN_MODELS_DIR", "/flare/NLDesignProtein/bryan/Diffusion-dev-space/models")
DATASETS_DIR = os.environ.get("PROTGEN_DATASETS_DIR", "/flare/NLDesignProtein/bryan/Diffusion-dev-space/datasets")
RUNS_DIR = os.environ.get("PROTGEN_RUNS_DIR", "/flare/NLDesignProtein/bryan/Diffusion-dev-space/runs")

# --- individual paths ---
SWISSPROT_CSV = os.environ.get("PROTGEN_SWISSPROT_CSV", f"{DATASETS_DIR}/fully_annotated_swiss_prot_080326.csv")
# Unannotated corpus = UniRef90, filtered to SwissProt's size constraints (30-500 aa) with exact
# SwissProt duplicates removed (see preprocess_fasta.py --exclude-csv). Packed *.bin + *.idx shards.
UNIREF90_FASTA = os.environ.get("PROTGEN_UNIREF90_FASTA", f"{DATASETS_DIR}/uniref90.fasta.gz")  # preprocess input
UNANNOTATED_SHARDS = os.environ.get("PROTGEN_UNANNOTATED_SHARDS", f"{DATASETS_DIR}/uniref90_shards")
TEXT_ENCODER = os.environ.get("PROTGEN_TEXT_ENCODER", f"{MODELS_DIR}/BioLinkBERT-base")       # PubMedBERT-full
BLOSUM_MAT = os.environ.get("PROTGEN_BLOSUM", f"{MODELS_DIR}/blosum62-special-MSA.mat")
TEXT_CACHE = os.environ.get("PROTGEN_TEXT_CACHE", f"{DATASETS_DIR}/swissprot_text_cache")    # precomputed BERT tokens
CKPT_DIR = os.environ.get("PROTGEN_CKPT_DIR", f"{RUNS_DIR}/checkpoints")


@dataclass
class DataCfg:
    max_residues: int = 500          # SwissProt build is filtered to 30-500 aa
    max_text_tokens: int = 128       # BiomedBERT caption cap (precompute + cross-attn/FiLIP)
    # STATIC SHAPES for XPU via length BUCKETS: each batch is padded to one of these fixed lengths and to
    # max_text_tokens, with a fixed composition -> a SMALL set of static shapes (one per bucket) that
    # IPEX/XPU compiles once each instead of recompiling per length. Per-batch size B = global_batch_tokens
    # // bucket_len (token budget), so short buckets pack more sequences at ~constant memory and the long
    # bucket stays memory-safe. The 1024 bucket is empty for the current 30-500 aa set but ready for later.
    length_buckets: tuple = (128, 256, 384, 512, 1024)
    # Length bucketing keeps the PAD tail short (PAD is modelled+attended for EOS-length).
    bucket_width: int = 32           # sequences batched within a length window of this size
    # Mixed corpus: mostly TrEMBL, but oversample SwissProt so the text/FiLIP pathway sees signal.
    # p_swissprot is the probability a training example is drawn from the labelled (text) corpus.
    p_swissprot: float = 0.25
    num_workers: int = 4
    prefetch_factor: int = 4


@dataclass
class OptCfg:
    global_batch_tokens: int = 32768     # target residues/step summed over the batch (bucketed)
    # sequences per rank per step. With mixed batches (~p_swissprot labelled), FiLIP negatives ≈
    # micro_batch * p_swissprot, so 64 gives ~16 labelled/batch (parity with the old homogeneous ~14)
    # and better tile utilization. Watch HBM on the next debug run; fall back to 32 if it OOMs.
    micro_batch: int = 64
    lr: float = 3e-4
    warmup_steps: int = 2000
    total_steps: int = 200_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # objective
    p_uncond: float = 0.15               # CFG dropout on labelled rows
    lam_filip: float = 0.2               # FiLIP auxiliary weight
    filip_max_rows: int = 16             # cap labelled rows in FiLIP -> bounds its (B_lab^2,L,T) HBM transient
    pad_loss_weight: float = 0.1         # down-weight PAD in OADM loss (fixed-512 canvas has a long PAD tail)
    beta: float = 0.5                    # hybrid substitution floor (low-corruption)
    beta_schedule: bool = True           # corruption-level beta (absorbing at high corruption)
    use_ipex: bool = True                # ipex.optimize: fused/faster but recompiles on new shapes (XPU)
    blosum_temp: float = 1.0             # BLOSUM substitution sharpness (inf -> uniform)
    # bookkeeping
    log_every: int = 50
    ckpt_every: int = 1000               # ~14min at scale; crashes are common on many tiles, so save often
    seed: int = 0


@dataclass
class RunCfg:
    device: str = "auto"                 # auto -> xpu on Aurora, cpu on a laptop
    data: DataCfg = field(default_factory=DataCfg)
    opt: OptCfg = field(default_factory=OptCfg)

    def model_config(self) -> ModelConfig:
        # Real vocab: 20 AA (ids 0..19) + EOS/PAD/MASK. ~55M params at these dims (see recurrent_oadm).
        return ModelConfig(
            vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
            d_model=512, n_heads=8, d_ff=1536,
            n_upstream=4, n_middle=8, n_downstream=4, n_recurrence=3,
            pb_layers=(1, 3, 5, 7), pb_dim=16, n_pb_heads=2,
            text_dim=768,                # BiomedBERT-base hidden
        )


CFG = RunCfg()

if __name__ == "__main__":
    print("REPO_ROOT     :", REPO_ROOT)
    print("SWISSPROT_CSV :", SWISSPROT_CSV, " exists:", os.path.exists(SWISSPROT_CSV))
    print("UNANNOT_SHARDS:", UNANNOTATED_SHARDS, " exists:", os.path.exists(UNANNOTATED_SHARDS))
    print("TEXT_ENCODER  :", TEXT_ENCODER, " exists:", os.path.exists(TEXT_ENCODER))
    print("BLOSUM_MAT    :", BLOSUM_MAT, " exists:", os.path.exists(BLOSUM_MAT))
    print("TEXT_CACHE    :", TEXT_CACHE, " exists:", os.path.exists(TEXT_CACHE))
    print("CKPT_DIR      :", CKPT_DIR)
    m = CFG.model_config()
    print(f"model         : d_model={m.d_model} layers={m.n_upstream}+{m.n_middle}(x{m.n_recurrence})+{m.n_downstream} "
          f"pb={m.pb_dim} vocab={m.vocab_size} (eos={m.eos_token_id} pad={m.pad_token_id} mask={m.mask_token_id})")
