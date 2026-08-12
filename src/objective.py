"""
Training objective for the recurrent OADM protein generator.

Two losses:
  1. OADM masked-ELBO (the generative objective). Order-agnostic absorbing-state discrete
     diffusion == any-order autoregressive. Per sequence: sample how many positions to mask,
     mask that many uniformly at random, and score the mean per-token NLL over the masked set.
  2. FiLIP token-level contrastive alignment (auxiliary, labelled rows only) — pulls the
     protein's own PB features toward the projected BiomedBERT description, residue-by-word.

Data handling:
  * A batch mixes labelled (SwissProt, has text) and unlabelled (TrEMBL, no text) rows.
  * `cond_mask` decides, per row, whether the *generative* pathway sees real text or the learned
    null token. Unlabelled rows are always null; labelled rows are null with prob `p_uncond`
    (classifier-free-guidance dropout).
  * FiLIP uses the real text for ALL labelled rows regardless of CFG dropout (it's a
    representation-alignment loss on z_pre, independent of the generative conditioning).

--------------------------------------------------------------------------------------------
OADM ELBO weighting (why "mean over masked, count ~ Uniform"):

  BioM3 / ARDM write, per sequence of length D,
        L = D * E_{t~U(1..D)} E_{sigma} [ 1/(D-t+1) * sum_{k in sigma(>=t)} -log p(x_k | x_sigma(<t)) ]
  With n := D-t+1 the number of masked positions, the inner term is
        (1/n) * sum_{masked} -log p   ==   mean NLL over the n masked positions.
  t ~ U(1..D)  <=>  n ~ U(1..D). The leading D is a constant (dropped for optimisation).
  So an unbiased per-sequence estimator is: n ~ U(1..D), mask n positions, take the MEAN NLL
  over them. Here D = number of real (non-PAD) residues; PAD is batch filler (no loss).
--------------------------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from .recurrent_oadm import RecurrentOADM, Config, filip_loss, count_params


# --------------------------------------------------------------------------------------
# OADM / hybrid corruption + loss
# --------------------------------------------------------------------------------------
def uniform_sub_probs(num_aa: int = 20) -> torch.Tensor:
    """Base substitution distribution P(j | i): uniform over the 19 non-identity amino acids.
    Swap in a BLOSUM-derived (row-normalised exchangeability) matrix later — same shape, same call."""
    m = torch.ones(num_aa, num_aa)
    m.fill_diagonal_(0.0)
    return m / m.sum(dim=1, keepdim=True)


def sample_corruption(tokens, target_mask, mask_id, beta: float = 1.0,
                      sub_probs: Optional[torch.Tensor] = None, num_aa: int = 20,
                      beta_schedule: bool = False):
    """
    Absorbing<->substitution hybrid forward corruption (D3PM family), parameterised by the
    mixing weight beta = P(corrupt -> MASK | position is corrupted):
        beta = 1  -> pure absorbing == EvoDiff-OADM (mask every corrupted position)
        beta = 0  -> pure substitution (D3PM); non-AA targets (EOS/PAD) always go to MASK
    Position selection is the SAME as OADM (n ~ U(1, D) per row), so the downstream loss
    (mean CE over corrupted positions) is unchanged across beta -- only the corruption differs.

    beta_schedule=True makes beta corruption-level dependent per row:
        beta_row = 1 - (1 - beta) * (1 - n_corrupt/D)
    i.e. pure absorbing when most of the sequence is corrupted (the GENERATION regime, which cold-
    starts from all-MASK and must stay OADM-clean) and substitution only near completion (the
    CORRECTION regime). `beta` then acts as the floor reached at low corruption. This preserves
    OADM cold-start generation while still training the AA->AA denoising the substitution corrector needs.

    Returns (corrupted, corrupt_pos); corrupt_pos are the positions to score (predict x0).
    """
    B, L = tokens.shape
    device = tokens.device
    n_targets = target_mask.sum(dim=1).clamp(min=1)
    u = torch.rand(B, device=device)
    n_corrupt = torch.minimum((u * n_targets.float()).floor().long() + 1, n_targets)
    scores = torch.rand(B, L, device=device).masked_fill(~target_mask, -1.0)
    ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)
    corrupt_pos = (ranks < n_corrupt[:, None]) & target_mask

    corrupted = tokens.clone()
    if beta >= 1.0 and not beta_schedule:                            # pure absorbing fast path == OADM
        corrupted[corrupt_pos] = mask_id
        return corrupted, corrupt_pos

    if beta_schedule:
        frac = (n_corrupt.float() / n_targets.float()).clamp(0.0, 1.0)   # (B,) corruption level
        beta_row = 1.0 - (1.0 - beta) * (1.0 - frac)                     # ->1 at high corruption
    else:
        beta_row = torch.full((B,), float(beta), device=device)

    if sub_probs is None:
        sub_probs = uniform_sub_probs(num_aa).to(device)
    is_aa = tokens < num_aa                                           # only amino acids can be substituted
    do_sub = corrupt_pos & is_aa & (torch.rand(B, L, device=device) >= beta_row[:, None])
    do_mask = corrupt_pos & ~do_sub                                  # MASK (includes all corrupted EOS/PAD)
    corrupted[do_mask] = mask_id
    if bool(do_sub.any()):
        x0 = tokens[do_sub]                                          # (M,) original residues
        corrupted[do_sub] = torch.multinomial(sub_probs[x0], 1).squeeze(1)
    return corrupted, corrupt_pos


def oadm_loss(logits: torch.Tensor, tokens: torch.Tensor, mask_pos: torch.Tensor) -> torch.Tensor:
    """Mean-over-masked NLL per sequence (the 1/n ELBO weight), then mean over the batch."""
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V).float(), tokens.reshape(-1),
                         reduction="none").reshape(B, L)
    ce = ce * mask_pos
    per_seq = ce.sum(dim=1) / mask_pos.sum(dim=1).clamp(min=1)
    return per_seq.mean()


# --------------------------------------------------------------------------------------
# Training step (labelled + unlabelled, CFG dropout, FiLIP)
# --------------------------------------------------------------------------------------
def training_step(model: RecurrentOADM, batch: dict,
                  p_uncond: float = 0.1, lam_filip: float = 0.1,
                  beta: float = 1.0, sub_probs: Optional[torch.Tensor] = None,
                  beta_schedule: bool = False):
    """
    batch:
      tokens:    (B, L) long
      labelled:  (B,) bool     — row has a real text annotation
      text_emb:  (B, T, text_dim) or None  — BiomedBERT token embeddings (garbage rows ok if not labelled)
      text_keep: (B, T) bool or None
    Returns (total_loss, metrics_dict).
    """
    cfg = model.cfg
    tokens = batch["tokens"]
    labelled = batch["labelled"]
    text_emb = batch.get("text_emb")
    text_keep = batch.get("text_keep")
    canvas_mask = batch.get("canvas_mask")                             # None -> whole width modelled
    B, L = tokens.shape
    device = tokens.device

    # Generative targets = every modelled canvas position (AA + EOS + PAD). EOS/PAD are predicted:
    # that is how the model learns to emit length. (canvas_mask excludes only true beyond-bucket filler.)
    target_mask = canvas_mask if canvas_mask is not None else torch.ones_like(tokens, dtype=torch.bool)

    # --- CFG dropout: which rows condition on real text this step ---
    keep_cond = torch.rand(B, device=device) > p_uncond
    cond_mask = labelled & keep_cond                                    # unlabelled never conditions

    # --- Forward corruption (absorbing<->substitution hybrid; beta=1 == OADM, same loss) ---
    corrupted, mask_pos = sample_corruption(tokens, target_mask, cfg.mask_token_id,
                                            beta=beta, sub_probs=sub_probs, beta_schedule=beta_schedule)

    logits, filip_pairs = model(corrupted, text_emb=text_emb, text_keep=text_keep,
                                cond_mask=cond_mask, canvas_mask=canvas_mask, collect_filip="last")

    L_oadm = oadm_loss(logits, tokens, mask_pos)

    # --- FiLIP over the labelled sub-batch (needs >= 2 for a contrastive signal) ---
    L_filip = tokens.new_zeros((), dtype=torch.float32)
    if lam_filip > 0 and text_emb is not None and int(labelled.sum()) >= 2 and filip_pairs:
        # Align text to true residues only -- exclude EOS/PAD (they are not part of the description).
        prot_keep = target_mask & (tokens != cfg.eos_token_id) & (tokens != cfg.pad_token_id)
        lab = labelled
        acc = 0.0
        for z_pre, zt_real in filip_pairs:                             # one pair per PB layer
            acc = acc + filip_loss(z_pre[lab], zt_real[lab], prot_keep[lab], text_keep[lab])
        L_filip = acc / len(filip_pairs)

    total = L_oadm + lam_filip * L_filip
    return total, {
        "total": float(total.detach()),
        "oadm": float(L_oadm.detach()),
        "filip": float(L_filip.detach()) if torch.is_tensor(L_filip) else float(L_filip),
        "n_cond": int(cond_mask.sum()),
        "n_labelled": int(labelled.sum()),
    }


# --------------------------------------------------------------------------------------
# Inference-time classifier-free guidance (applied per decoding step on the masked-position logits)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def cfg_guided_logits(model: RecurrentOADM, tokens, text_emb, text_keep, w: float = 3.0):
    """logits = uncond + w*(cond - uncond). w=0 -> unconditional, w=1 -> plain conditional,
    w>1 -> amplified text adherence. In the sampler this is called each step on the current canvas."""
    B = tokens.shape[0]
    cond, _ = model(tokens, text_emb=text_emb, text_keep=text_keep,
                    cond_mask=torch.ones(B, dtype=torch.bool, device=tokens.device),
                    collect_filip=None)
    uncond, _ = model(tokens, text_emb=None, collect_filip=None)       # learned null token
    return uncond + w * (cond - uncond)


# --------------------------------------------------------------------------------------
# Overfit demo: watch the loss fall and the conditioning gates open on a tiny fixed batch.
# --------------------------------------------------------------------------------------
def _gate_report(model: RecurrentOADM):
    g_up, g_cross = [], []
    for l in model.mid_layers:
        if hasattr(l, "pb"):
            g_up.append(torch.tanh(l.pb.g_up).item())
            g_cross.append(torch.tanh(l.pb.g_cross).item())
    reinj = torch.tanh(model.reinject_gate).item()
    return sum(g_up) / len(g_up), sum(g_cross) / len(g_cross), reinj


def _cfg_delta(model, tokens, text_emb, text_keep):
    with torch.no_grad():
        B = tokens.shape[0]
        cond, _ = model(tokens, text_emb=text_emb, text_keep=text_keep,
                        cond_mask=torch.ones(B, dtype=torch.bool), collect_filip=None)
        uncond, _ = model(tokens, text_emb=None, collect_filip=None)
    return (cond - uncond).float().norm().item()


def _eval_oadm(model, tokens, text_emb, text_keep, cfg, frac=0.5):
    """Deterministic mask of each row's C-terminal `frac`; report OADM loss with and without text.
    Low-variance progress signal: conditional should fall toward 0 (text identifies the row);
    unconditional falls more slowly (must memorise from the N-terminal context alone)."""
    target = tokens != cfg.pad_token_id
    mask_pos = torch.zeros_like(target)
    for i in range(tokens.shape[0]):
        idx = target[i].nonzero(as_tuple=True)[0]
        k = max(1, int(len(idx) * frac))
        mask_pos[i, idx[-k:]] = True
    corrupted = torch.where(mask_pos, torch.full_like(tokens, cfg.mask_token_id), tokens)
    B = tokens.shape[0]
    model.eval()
    with torch.no_grad():
        cond_logits, _ = model(corrupted, text_emb=text_emb, text_keep=text_keep,
                               cond_mask=torch.ones(B, dtype=torch.bool), collect_filip=None)
        unc_logits, _ = model(corrupted, text_emb=None, collect_filip=None)
        l_cond = oadm_loss(cond_logits, tokens, mask_pos).item()
        l_unc = oadm_loss(unc_logits, tokens, mask_pos).item()
    model.train()
    return l_cond, l_unc


if __name__ == "__main__":
    torch.manual_seed(0)

    # Small, fast config for a CPU overfit sanity check.
    cfg = Config(
        vocab_size=25, eos_token_id=22, pad_token_id=23, mask_token_id=24,
        d_model=128, n_heads=4, d_ff=384,
        n_upstream=2, n_middle=4, n_downstream=2, n_recurrence=2,
        pb_layers=(1, 3), pb_dim=16, n_pb_heads=2, text_dim=64,
    )
    model = RecurrentOADM(cfg)
    print(f"demo params={count_params(model)/1e6:.2f}M")

    B, L, T = 4, 18, 6
    lengths = [14, 11, 9, 6]                          # residue counts; then EOS; then modelled PAD
    tokens = torch.full((B, L), cfg.pad_token_id, dtype=torch.long)
    for i, n in enumerate(lengths):
        tokens[i, :n] = torch.randint(0, 20, (n,))   # AA ids 0..19
        tokens[i, n] = cfg.eos_token_id              # EOS marks the boundary (length = n)
    labelled = torch.tensor([True, True, False, False])
    text_emb = torch.randn(B, T, cfg.text_dim)       # fixed random "annotations"
    text_keep = torch.ones(B, T, dtype=torch.bool)
    batch = {"tokens": tokens, "labelled": labelled, "text_emb": text_emb, "text_keep": text_keep}

    ec, eu = _eval_oadm(model, tokens, text_emb, text_keep, cfg)
    print(f"init: eval oadm cond={ec:.3f} uncond={eu:.3f} | cfg_delta="
          f"{_cfg_delta(model, tokens, text_emb, text_keep):.4f} | gates(up,cross,reinj)={_gate_report(model)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, 601):
        opt.zero_grad()
        loss, m = training_step(model, batch, p_uncond=0.1, lam_filip=0.3)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            gu, gc, gr = _gate_report(model)
            ec, eu = _eval_oadm(model, tokens, text_emb, text_keep, cfg)
            print(f"step {step:3d} | eval oadm cond={ec:.3f} uncond={eu:.3f} | filip {m['filip']:.3f} | "
                  f"gates up={gu:+.3f} cross={gc:+.3f} reinj={gr:+.3f} | "
                  f"cfg_delta={_cfg_delta(model, tokens, text_emb, text_keep):.2f}")

    # Length check: reveal the residues, mask from the boundary on, ask where EOS lands.
    model.eval()
    with torch.no_grad():
        probe = tokens.clone()
        for i, n in enumerate(lengths):
            probe[i, n:] = cfg.mask_token_id                    # mask EOS + everything after
        lg, _ = model(probe, text_emb=text_emb, text_keep=text_keep,
                      cond_mask=torch.ones(B, dtype=torch.bool), collect_filip=None)
        pred = lg.argmax(-1)
        eos_at = [int(pred[i, n] == cfg.eos_token_id) for i, n in enumerate(lengths)]
    print(f"EOS predicted at true boundary? {eos_at} (1=yes) for lengths {lengths}")
    print("Expect: conditional OADM loss falls toward ~0 (text lets labelled rows nail their "
          "C-terminus), uncond falls more slowly, FiLIP -> ~0, gates open (now honest), cfg_delta grows, "
          "and EOS lands at the true length.")
