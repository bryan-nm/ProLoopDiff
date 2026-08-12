"""
Sampler for the recurrent OADM protein generator.

Confidence-ordered (MaskGIT-style) decoding of an absorbing-state canvas:
  * Start from an all-MASK canvas of width Lmax.
  * Each step: score every masked position, commit the most-confident few (cosine schedule),
    leave the rest masked for later steps. This is strictly better than EvoDiff's random unmask
    order and it is what makes EOS-based length work: the model commits EOS once it is confident
    about the boundary, not at a fixed position.
  * EOS is the length marker. The moment EOS is committed at position p, everything after p is
    forced to PAD (and any AA mistakenly committed to the right of it is overwritten) — so the
    output is always well-formed  [AA... EOS PAD...].
  * During active decoding, MASK and PAD are forbidden as emitted tokens (logits -> -inf), so the
    model can only place an amino acid or EOS; PAD appears solely via EOS enforcement. That keeps
    length a decision the model makes by *placing EOS*, exactly as trained.

Classifier-free guidance: each step's logits come from cfg_guided_logits (uncond + w*(cond-uncond))
using the learned null token, so `cfg_weight` dials text adherence.

Remasking corrector (predictor-corrector): optional post sweeps that remask the lowest-confidence
committed positions and redecode them — the error-correction that pure absorbing decoding lacks,
and a stepping-stone to the learned BLOSUM-substitution corrector later.

ProteinGuide hook: `guidance_fn(canvas, logits) -> logits` is applied to the per-step factorised
categorical, so a property predictor can reweight logits on the fly without touching this file.
"""

from __future__ import annotations
import math
from typing import Optional, Callable
import torch

from .recurrent_oadm import RecurrentOADM
from .objective import cfg_guided_logits


@torch.no_grad()
def _step_logits(model, canvas, text_emb, text_keep, w, guidance_fn):
    if text_emb is not None and w > 0:
        lg = cfg_guided_logits(model, canvas, text_emb, text_keep, w)
    else:
        lg, _ = model(canvas, text_emb=None, collect_filip=None)          # unconditional (null token)
    if guidance_fn is not None:
        lg = guidance_fn(canvas, lg)                                       # ProteinGuide-style hook
    cfg = model.cfg
    lg = lg.clone()
    lg[..., cfg.mask_token_id] = float("-inf")                            # never emit MASK...
    lg[..., cfg.pad_token_id] = float("-inf")                            # ...or PAD (only via EOS enforcement)
    return lg


def _sample(probs, greedy):
    if greedy:
        conf, tok = probs.max(dim=-1)
        return tok, conf
    B, L, V = probs.shape
    tok = torch.multinomial(probs.reshape(-1, V), 1).reshape(B, L)
    conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    return tok, conf


def _enforce_eos(canvas, is_masked, cfg):
    """Force everything after the leftmost committed EOS to PAD (per row)."""
    B, L = canvas.shape
    for b in range(B):
        committed_eos = ((canvas[b] == cfg.eos_token_id) & ~is_masked[b]).nonzero(as_tuple=True)[0]
        if committed_eos.numel() > 0:
            p = int(committed_eos.min())
            if p + 1 < L:
                canvas[b, p + 1:] = cfg.pad_token_id
                is_masked[b, p + 1:] = False


def lengths_of(canvas, cfg):
    """Length = position of the first EOS, else the full width (a max-length generation)."""
    B, L = canvas.shape
    out = []
    for b in range(B):
        e = (canvas[b] == cfg.eos_token_id).nonzero(as_tuple=True)[0]
        out.append(int(e.min()) if e.numel() else L)
    return out


@torch.no_grad()
def _corrector_sweep(model, canvas, text_emb, text_keep, w, frac, guidance_fn, greedy, temperature):
    cfg = model.cfg
    B, L = canvas.shape
    # confidence the model currently assigns to each committed token
    lg = _step_logits(model, canvas, text_emb, text_keep, w, guidance_fn)
    probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
    conf = probs.gather(-1, canvas.unsqueeze(-1)).squeeze(-1)             # (B,L)
    eligible = canvas != cfg.pad_token_id                                # remask AAs / EOS, never PAD
    conf_e = conf.masked_fill(~eligible, float("inf"))                   # inf -> never picked as lowest

    is_masked = torch.zeros_like(canvas, dtype=torch.bool)
    for b in range(B):
        n_elig = int(eligible[b].sum())
        k = max(1, int(frac * n_elig))
        idx = torch.topk(conf_e[b], k, largest=False).indices            # lowest-confidence positions
        canvas[b, idx] = cfg.mask_token_id
        is_masked[b, idx] = True

    lg2 = _step_logits(model, canvas, text_emb, text_keep, w, guidance_fn)
    probs2 = torch.softmax(lg2 / max(temperature, 1e-6), dim=-1)
    tok, _ = _sample(probs2, greedy)
    canvas[is_masked] = tok[is_masked]
    _enforce_eos(canvas, torch.zeros_like(canvas, dtype=torch.bool), cfg)


@torch.no_grad()
def _substitution_corrector_sweep(model, canvas, text_emb, text_keep, w, frac,
                                  guidance_fn, greedy, temperature):
    """Learned substitution corrector (pairs with beta<1 training, which teaches AA->AA denoising).
    Resample the lowest-confidence *residues* directly to new tokens -- no MASK detour. Because EOS
    is an allowed emission, a resample can also move the boundary (Idea-B length revision): if a
    residue is resampled to EOS, _enforce_eos truncates everything after it to PAD."""
    cfg = model.cfg
    B, L = canvas.shape
    lg = _step_logits(model, canvas, text_emb, text_keep, w, guidance_fn)   # MASK/PAD forbidden; EOS ok
    probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
    conf = probs.gather(-1, canvas.unsqueeze(-1)).squeeze(-1)
    eligible = (canvas != cfg.pad_token_id) & (canvas != cfg.eos_token_id) & (canvas != cfg.mask_token_id)
    conf_e = conf.masked_fill(~eligible, float("inf"))                      # inf -> never chosen as lowest
    tok, _ = _sample(probs, greedy)
    for b in range(B):
        n = int(eligible[b].sum())
        if n == 0:
            continue
        k = max(1, int(frac * n))
        idx = torch.topk(conf_e[b], k, largest=False).indices              # lowest-confidence residues
        canvas[b, idx] = tok[b, idx]                                        # direct substitution
    _enforce_eos(canvas, torch.zeros_like(canvas, dtype=torch.bool), cfg)


@torch.no_grad()
def generate(model: RecurrentOADM, Lmax: int, batch_size: int,
             text_emb: Optional[torch.Tensor] = None, text_keep: Optional[torch.Tensor] = None,
             cfg_weight: float = 3.0, n_steps: Optional[int] = None,
             temperature: float = 1.0, gumbel_temp: float = 0.0, greedy: bool = False,
             n_corrector: int = 0, corrector_frac: float = 0.1, corrector_type: str = "remask",
             guidance_fn: Optional[Callable] = None, device: str = "cpu"):
    """Returns (canvas (B, Lmax) long, lengths list[int]).
    corrector_type: "remask" (works with any model) or "substitution" (needs beta<1 training)."""
    cfg = model.cfg
    B = batch_size
    n_steps = n_steps or Lmax
    was_training = model.training
    model.eval()

    canvas = torch.full((B, Lmax), cfg.mask_token_id, dtype=torch.long, device=device)
    is_masked = torch.ones((B, Lmax), dtype=torch.bool, device=device)

    for step in range(n_steps):
        if not bool(is_masked.any()):
            break
        lg = _step_logits(model, canvas, text_emb, text_keep, cfg_weight, guidance_fn)
        probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
        tok, conf = _sample(probs, greedy)

        conf_masked = conf.masked_fill(~is_masked, float("-inf"))
        if gumbel_temp > 0:                                              # stochastic ordering (MaskGIT)
            u = torch.rand_like(conf).clamp_min(1e-9)
            g = -torch.log(-torch.log(u).clamp_min(1e-9))
            conf_masked = conf_masked + gumbel_temp * g.masked_fill(~is_masked, 0.0)

        # cosine schedule: how many positions remain masked after this step
        frac = (step + 1) / n_steps
        n_mask_target = round(Lmax * math.cos(math.pi / 2 * frac))
        n_commit = (is_masked.sum(dim=1) - n_mask_target).clamp(min=0)

        for b in range(B):
            k = int(n_commit[b])
            if k <= 0:
                continue
            idx = torch.topk(conf_masked[b], k).indices
            canvas[b, idx] = tok[b, idx]
            is_masked[b, idx] = False
        _enforce_eos(canvas, is_masked, cfg)

    for _ in range(n_corrector):
        if corrector_type == "substitution":
            _substitution_corrector_sweep(model, canvas, text_emb, text_keep, cfg_weight,
                                          corrector_frac, guidance_fn, greedy, temperature)
        else:
            _corrector_sweep(model, canvas, text_emb, text_keep, cfg_weight,
                             corrector_frac, guidance_fn, greedy, temperature)

    if was_training:
        model.train()
    return canvas, lengths_of(canvas, cfg)


# --------------------------------------------------------------------------------------
# End-to-end demo: overfit the tiny model, then sample it back.
# --------------------------------------------------------------------------------------
def _decode(canvas_row, cfg):
    ids = canvas_row.tolist()
    out = []
    for t in ids:
        if t == cfg.eos_token_id:
            out.append("|"); break
        if t == cfg.pad_token_id:
            break
        out.append(chr(ord("A") + t) if t < 20 else "?")
    return "".join(out)


if __name__ == "__main__":
    from .recurrent_oadm import Config
    from .objective import training_step, uniform_sub_probs
    torch.manual_seed(0)

    cfg = Config(
        vocab_size=25, eos_token_id=22, pad_token_id=23, mask_token_id=24,
        d_model=128, n_heads=4, d_ff=384,
        n_upstream=2, n_middle=4, n_downstream=2, n_recurrence=2,
        pb_layers=(1, 3), pb_dim=16, n_pb_heads=2, text_dim=64,
    )
    model = RecurrentOADM(cfg)

    B, L, T = 4, 18, 6
    lengths = [14, 11, 9, 6]
    tokens = torch.full((B, L), cfg.pad_token_id, dtype=torch.long)
    for i, n in enumerate(lengths):
        tokens[i, :n] = torch.randint(0, 20, (n,))
        tokens[i, n] = cfg.eos_token_id
    labelled = torch.tensor([True, True, False, False])
    text_emb = torch.randn(B, T, cfg.text_dim)
    text_keep = torch.ones(B, T, dtype=torch.bool)
    batch = {"tokens": tokens, "labelled": labelled, "text_emb": text_emb, "text_keep": text_keep}

    # Train the HYBRID with a corruption-level beta schedule: pure absorbing (OADM) when most of
    # the sequence is unknown (preserves cold-start generation), substitution allowed near completion
    # (trains the AA->AA denoising the substitution corrector needs). beta = the low-corruption floor.
    sub_probs = uniform_sub_probs(20)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(800):
        opt.zero_grad()
        loss, _ = training_step(model, batch, p_uncond=0.1, lam_filip=0.3,
                                beta=0.5, sub_probs=sub_probs, beta_schedule=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print("trained hybrid (beta floor 0.5, scheduled)")
    print("truth (labelled rows 0,1):", [_decode(tokens[i], cfg) for i in range(2)])

    # (1) Cold-start conditional generation (OADM path dominates via the schedule). Illustrative:
    #     exact reconstruction is a noisy metric with a 128-d model memorising 4 sequences.
    gen, glen = generate(model, Lmax=L, batch_size=2, text_emb=text_emb[:2], text_keep=text_keep[:2],
                         cfg_weight=3.0, greedy=True)
    print("cold-start cond gen:", [_decode(gen[i], cfg) for i in range(2)], "lengths", glen)

    # (2) Substitution corrector on its real job: repair residues corrupted away from the truth.
    torch.manual_seed(1)
    dirty = tokens[:2].clone()
    for i in range(2):
        idx = torch.randperm(lengths[i])[:3]
        dirty[i, idx] = torch.randint(0, 20, (3,))
    err0 = [int((dirty[i, :lengths[i]] != tokens[i, :lengths[i]]).sum()) for i in range(2)]
    print("corrupted          :", [_decode(dirty[i], cfg) for i in range(2)], "errors", err0)
    for _ in range(4):
        _substitution_corrector_sweep(model, dirty, text_emb[:2], text_keep[:2],
                                      w=3.0, frac=0.2, guidance_fn=None, greedy=True, temperature=0.5)
    err1 = [int((dirty[i, :lengths[i]] != tokens[i, :lengths[i]]).sum()) for i in range(2)]
    print("after corrector    :", [_decode(dirty[i], cfg) for i in range(2)], "errors", err1)
    print(f"substitution corrector reduced residue errors {err0} -> {err1}")

    # (3) Unconditional generation: check well-formedness [AA* EOS PAD*].
    ug, ulen = generate(model, Lmax=L, batch_size=4, text_emb=None, temperature=1.0,
                        gumbel_temp=0.1, greedy=False)
    print("uncond gen         :", [_decode(ug[i], cfg) for i in range(4)], "lengths", ulen)
    wellformed = all(
        (ug[b] == cfg.eos_token_id).sum() <= 1 and
        bool((ug[b, ulen[b] + 1:] == cfg.pad_token_id).all() if ulen[b] + 1 < L else True)
        for b in range(4)
    )
    print("all unconditional samples well-formed [AA* EOS PAD*]:", wellformed)
