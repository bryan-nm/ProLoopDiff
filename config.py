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
# ESMFold2-Fast weights for the structural eval (pLDDT). NOT under MODELS_DIR: this is the path
# the EsmFold repo's speed_test.pbs uses and where the weights are already staged. The ESM-C 6B
# backbone named in its config is resolved from the HuggingFace cache -- pre-cache it from a login
# node and set HF_HUB_OFFLINE=1 on compute nodes. See the EsmFold repo's README.
ESMFOLD_WEIGHTS = os.environ.get("PROTGEN_ESMFOLD_WEIGHTS",
                                 "/flare/NLDesignProtein/bryan/models/ESMFold2-Fast")


@dataclass
class DataCfg:
    max_text_tokens: int = 128       # BiomedBERT caption cap (precompute + PB cross-attention)
    # STATIC SHAPES for XPU via length BUCKETS: each batch is padded to one of these fixed lengths and to
    # max_text_tokens, with a fixed composition -> a SMALL set of static shapes (one per bucket) that
    # IPEX/XPU compiles once each instead of recompiling per length. Per-batch size B = global_batch_tokens
    # // bucket_len (token budget), so short buckets pack more sequences at ~constant memory and the long
    # bucket stays memory-safe. Both corpora are built to 30-500 aa, so the 1024 bucket is empty for the
    # current data (it would halve B and roughly double attention HBM) but is ready for later.
    # Length bucketing also keeps the PAD tail short (PAD is modelled+attended for EOS-length).
    length_buckets: tuple = (512,)
    # Mixed corpus: mostly TrEMBL, but oversample SwissProt so the text pathway sees signal.
    # p_swissprot is the probability a training example is drawn from the labelled (text) corpus.
    p_swissprot: float = 0.25
    num_workers: int = 4
    prefetch_factor: int = 4         # batches prefetched per worker (only used when num_workers > 0)


@dataclass
class OptCfg:
    # Per-rank token budget per step; the sampler derives each bucket's batch size as
    # global_batch_tokens // bucket_len (so B is 256 at L=128 down to 64 at L=512).
    global_batch_tokens: int = 32768
    lr: float = 3e-4
    warmup_steps: int = 2000
    total_steps: int = 200_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # A non-finite gradient norm makes clip_grad_norm_'s coefficient NaN, which would turn every
    # parameter NaN on the next opt.step(). The trainer skips those updates instead; this many
    # CONSECUTIVE skips means something is structurally wrong, so abort rather than spin.
    max_skip_streak: int = 20
    # objective
    p_uncond: float = 0.15               # CFG dropout on labelled rows
    pad_loss_weight: float = 0.1         # down-weight PAD in OADM loss (fixed-512 canvas has a long PAD tail)
    beta: float = 0.5                    # hybrid substitution floor (low-corruption)
    beta_schedule: bool = True           # corruption-level beta (absorbing at high corruption)
    use_ipex: bool = True                # ipex.optimize: fused/faster but recompiles on new shapes (XPU)
    blosum_temp: float = 1.0             # BLOSUM substitution sharpness (inf -> uniform)
    # --- periodic generative eval (unconditional sampling) ---
    # The OADM loss says nothing about whether the model places EOS sensibly, so sample and measure.
    # Cost, in training-step equivalents: eval_steps * (eval_n * eval_canvas) / (3 * global_batch_tokens)
    # -- one training step is fwd+bwd (~3 forward-units) over global_batch_tokens. At 64/100/512 that
    # is ~33 steps, so eval_every=1000 predicts ~3.3% of wall time (the trainer measures and logs the
    # real figure, which runs a little higher: eval is all at L=512 while training averages over the
    # buckets at constant token budget, and attention is quadratic in L).
    # eval_every=100 would only afford ~9 decoding steps at this width -- ~57 tokens committed per
    # step, which measures the decode schedule rather than the model. 1000 also aligns with
    # ckpt_every, so every checkpoint gets a length statistic next to it.
    eval_every: int = 1000               # 0 disables
    eval_n: int = 100                    # sequences per eval
    eval_canvas: int = 512               # largest training bucket = the model's widest length prior
    eval_steps: int = 64                 # confidence-ordered decoding steps
    # --- structural eval (ESMFold2-Fast pLDDT), run by the dedicated eval node ---
    # Folding is the expensive metric: ~1.1 s/sequence on one PVC tile at fold_steps=20, and
    # sequences are scored one at a time (no intra-batch parallelism). fold_n is PER RANK and every
    # rank folds concurrently on its own tile, so 100/rank is ~110s wall-clock and yields
    # 100*world sequences of statistics. 0 disables folding.
    fold_n: int = 100                    # sequences folded per rank per eval round
    fold_min_len: int = 10               # skip shorter generations; too short to fold meaningfully
    # DO NOT lower fold_steps below 20: pLDDT does not degrade gracefully, it collapses to the
    # ~0.25 no-information floor between 10 and 20 steps, which would silently look like a metric.
    fold_steps: int = 20                 # ESMFold2 diffusion sampling steps
    fold_loops: int = 1                  # trunk recycling; measured to not move pLDDT at all
    plddt_confident: float = 0.70        # "confident fold" threshold (0-1 scale; = 70 on AF's 0-100)
    # pTM is global (is the OVERALL topology right) where pLDDT is local (is each residue placed
    # confidently). They come apart: a chain of well-formed helices floating in the wrong
    # arrangement scores high pLDDT and low pTM, so watching only pLDDT can miss that the model
    # makes good secondary structure and no real fold. 0.5 is the usual "topology likely correct"
    # line. Reported on the native 0-1 scale (unlike pLDDT, pTM has no 0-100 convention).
    ptm_confident: float = 0.50
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
