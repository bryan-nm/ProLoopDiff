# protein_generation

A sequence-only, text-conditionable protein generator: an any-order autoregressive / absorbing-state
discrete diffusion model (EvoDiff-OADM style) on a **looped transformer** trunk, with a
low-dimensional **privileged-basis** conditioning interface that keeps the model interpretable and
[ProteinGuide](https://www.nature.com/articles/s41587-026-03207-z)-compatible. Trained on sequence
only; text is an optional cross-attention pathway that can be swapped for other modalities later.

## Design at a glance

- **Backbone** — 4 upstream + 8 middle (distinct layers, looped `n_recurrence` times) + 4 downstream,
  bidirectional (OADM is not causal), RoPE on self-attention Q/K only.
- **Conditioning** — a subset of middle layers ("PB layers") project the residual stream into a 16-d
  privileged-basis subspace where frozen-BiomedBERT text is written in via zero-init **gated** cross-attention
  (affine-free RMS norm on the writeback makes the gate the honest throttle). A learned **null token** makes
  the unconditional pass a calibrated CFG baseline. A **FiLIP** head aligns each PB layer's residue features
  with the text token-by-token.
- **Length** — the model emits an **EOS** token; its position is the length. PAD after EOS is modelled and
  attended (this is what makes emergent-EOS length work).
- **Objective** — x0-denoising cross-entropy over a hybrid absorbing↔substitution corruption, parameterised
  by mixing weight **β** (β=1 is exact OADM). A **β-schedule** keeps corruption pure-absorbing when most of the
  sequence is unknown (generation regime) and allows BLOSUM-weighted substitution only near completion
  (correction regime) — otherwise cold-start generation degrades (EvoDiff's OADM>D3PM effect).
- **Sampling** — confidence-ordered (MaskGIT) decoding with per-step CFG, EOS enforcement, and a choice of
  remask or **substitution** corrector; a `guidance_fn` hook for on-the-fly ProteinGuide property steering.

See the memory notes / design docs for the full rationale.

## Layout

```
config.py            # owns ALL paths + model/data/opt config (python config.py prints resolved paths)
src/
  recurrent_oadm.py  # model: looped transformer + PB cross-attention + FiLIP head
  objective.py       # hybrid corruption + OADM/D3PM loss + FiLIP + CFG training_step
  sampler.py         # confidence-ordered decoding + CFG + correctors
  blosum.py          # EvoDiff BLOSUM62 -> substitution matrix
  data.py            # tokenizer, text embedders, TrEMBL shards, mixed dataset, bucketing, collate
  dist.py            # Aurora XPU + oneCCL bootstrap (one rank per tile)
  train.py           # training loop (ipex.optimize, bf16, manual data-parallel)
  preprocess_fasta.py# TrEMBL FASTA -> packed uint8 shards
scripts/
  pbs_common.sh      # shared Aurora job setup (module load frameworks, device selector, CCL env)
  train.pbs          # qsub entry point
```

## Setup

Local (workstation smoke tests) — reuse the sibling `mini-embed-filip` conda env or:
```bash
conda create -n protein-gen python=3.11 && conda activate protein-gen
pip install torch numpy transformers safetensors
```
Aurora — `module load frameworks` provides torch + IPEX + oneCCL; create the venv referenced in
`scripts/pbs_common.sh` (`--system-site-packages`).

## Data

- **SwissProt** (labelled): a 3-column CSV `primary_Accession, protein_sequence, [final]text_caption`,
  filtered to non-fragment, 30–500 aa. Point `PROTGEN_SWISSPROT_CSV` at it.
- **UniRef90** (unlabelled): the base corpus. Filter to SwissProt's size window (30–500 aa) and remove
  sequences that exactly match a SwissProt entry, then pre-tokenise to packed shards (do NOT stream FASTA
  in the loop):
  ```bash
  python -m src.preprocess_fasta uniref90.fasta.gz $PROTGEN_UNANNOTATED_SHARDS \
      --min 30 --max 500 --dedup --exclude-csv $PROTGEN_SWISSPROT_CSV
  ```
- **Text cache** (labelled): precompute BiomedBERT token embeddings for the SwissProt captions ONCE, so
  BERT never runs in the training loop. `qsub scripts/precompute_text.pbs` (or `python -m src.precompute_text`).
  The trainer auto-uses `config.TEXT_CACHE` when its `fingerprint.json` is present.

## Run

```bash
# resolve + print every path a run will use
python config.py

# local smoke: real SwissProt CSV, dummy text embeddings, a few CPU steps
PROTGEN_SWISSPROT_CSV=/path/to/swissprot.csv \
PROTGEN_BLOSUM=/path/to/blosum62-special-MSA.mat \
python -m src.train --smoke

# Aurora
qsub scripts/train.pbs
```

## Status / TODO

- **Working & tested locally**: model, objective (β=1 OADM ↔ β<1 hybrid), sampler (+ substitution corrector),
  BLOSUM, full data pipeline, training loop (loss descends on real SwissProt), UniRef90 shard round-trip,
  BiomedBERT text **precompute → cache** (byte-verified: cached embeddings match live to fp16, accessions align).
- **Next**: prepare the real corpora (UniRef90 shards + text cache), then the first real Aurora run.
- **Optional/later**: tight D3PM KL-ELBO for β<1 (surrogate x0-CE is exact at β=1 and standard for β<1);
  checkpoint resume.
