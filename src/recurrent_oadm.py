"""
Recurrent OADM protein generator — architecture sketch.

Sequence-only, any-order autoregressive / absorbing-state discrete diffusion (EvoDiff-OADM
style) with a looped ("distinct-8-with-loop") transformer trunk and a text-conditioning
pathway that lives in a low-dimensional *privileged-basis* (PB) subspace.

Design commitments encoded here (see conversation):
  * Trunk = 4 upstream + 8 distinct middle (shared across N recurrence passes) + 4 downstream.
  * PB layers = a subset of the middle layers (default every other). Text cross-attention and
    a FiLIP-style late-interaction head both operate in the SAME 16-d PB subspace, so the
    conditioning entry point and the interpretable subspace coincide.
  * Everything is CFG-ready: pass text_emb=None for the unconditional (null) branch. All
    conditioning enters through zero-initialised gates, so at init the model == a clean
    unconditional OADM and the gates learn how much to open.
  * ProteinGuide-compatible by construction: the model exposes one factorised per-position
    categorical head over the AA vocab; all the recurrence / bottleneck / cross-attn is
    internal and invisible to guidance, which only ever touches output logits.

(1) Aurora / Intel XPU. Code is device-agnostic and avoids CUDA-only paths (no flash-attn
    import, no apex/triton). Attention uses torch.nn.functional.scaled_dot_product_attention,
    which dispatches to an oneDNN/efficient kernel on XPU via Intel Extension for PyTorch.
    For a real run:  import intel_extension_for_pytorch as ipex; model = model.to('xpu');
    model = ipex.optimize(model, dtype=torch.bfloat16); autocast(device_type='xpu', bf16);
    distributed via the 'ccl' backend (oneCCL). Those live in the trainer, not here.

(2) RoPE vs the privileged basis. RoPE is applied ONLY to the Q/K of the *self-attention*
    heads, i.e. inside the attention head subspace. It never rotates the residual stream, and
    in particular it never touches the PB down-projection (which reads the un-rotated residual
    stream) or the 16-d PB subspace. Text cross-attention carries NO RoPE. So the PB basis is
    defined in a space RoPE provably does not mix. This separation is the whole point — see
    SelfAttention (RoPE here) vs PBConditioning (no RoPE, ever).

(4) FiLIP head compares, token-by-token, the protein's own PB features against a learned
    projection of BiomedBERT token embeddings into the same 16-d space — a fine-grained
    (residue x word) late-interaction contrastive signal, usable both as an auxiliary training
    loss on the annotated (SwissProt) sub-batch and as an interpretability probe.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # vocab: 20 aa + a few specials (mask, pad, eos, unk, bos, gap...). EvoDiff uses ~31.
    vocab_size: int = 33
    eos_token_id: int = 30    # length marker: the model predicts EOS; its position = sequence length
    pad_token_id: int = 31    # post-EOS filler; MODELLED and attended (predicted as PAD), not masked out
    mask_token_id: int = 32

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1536              # SwiGLU inner dim (~8/3 * d_model → param-parity with 4x GELU)

    n_upstream: int = 4
    n_middle: int = 8             # distinct middle layers, shared across recurrence passes
    n_downstream: int = 4
    n_recurrence: int = 3         # passes through the middle stack (adaptive-compute knob)

    pb_layers: tuple = (1, 3, 5, 7)   # which middle layers are privileged-basis / text-conditioned
    pb_dim: int = 16                  # dimensionality of the PB / cross-attn / FiLIP subspace
    n_pb_heads: int = 2               # heads for the in-PB cross-attention (16/2 = 8-d heads)

    text_dim: int = 768           # BiomedBERT-full hidden size (token-level embeddings)

    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True


# --------------------------------------------------------------------------------------
# Norm + FFN
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.affine = affine
        self.w = nn.Parameter(torch.ones(d)) if affine else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(dt)
        return self.w * x if self.affine else x


class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------------------
# RoPE  (self-attention Q/K only; residual stream & PB subspace never rotated)
# --------------------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)            # (L, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (L, hd)  split-half convention
        cos = emb.cos()[None, None, :, :].to(dtype)      # (1,1,L,hd)
        sin = emb.sin()[None, None, :, :].to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, H, L, hd) ; cos, sin: (1, 1, L, hd)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# --------------------------------------------------------------------------------------
# Bidirectional self-attention (OADM denoiser is NOT causal)
# --------------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        assert self.h * self.hd == cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.hd, cfg.rope_base)
        self.drop_p = cfg.dropout

    def forward(self, x: torch.Tensor, keep_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # (B, H, L, hd)

        # --- RoPE applied HERE ONLY: rotates the attention head subspace, not the residual
        #     stream and not the PB subspace. This is the (2) guarantee. ---
        cos, sin = self.rope(L, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        attn_mask = None
        if keep_mask is not None:                        # keep_mask: (B, L) True = real token
            attn_mask = keep_mask[:, None, None, :]       # (B,1,1,L) bool; True = allowed to attend
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o(out)


# --------------------------------------------------------------------------------------
# In-PB text cross-attention (16-d; no RoPE)
# --------------------------------------------------------------------------------------
class PBCrossAttention(nn.Module):
    """Query = protein PB features (B,L,p); Key/Value = text projected into the SAME p-dim."""
    def __init__(self, cfg: Config):
        super().__init__()
        p = cfg.pb_dim
        self.h = cfg.n_pb_heads
        self.hd = p // self.h
        assert self.h * self.hd == p
        self.q = nn.Linear(p, p, bias=False)
        self.k = nn.Linear(p, p, bias=False)
        self.v = nn.Linear(p, p, bias=False)
        self.o = nn.Linear(p, p, bias=False)

    def forward(self, z: torch.Tensor, zt: torch.Tensor,
                text_keep: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, p = z.shape
        T = zt.shape[1]
        q = self.q(z).view(B, L, self.h, self.hd).transpose(1, 2)     # (B,h,L,hd)
        k = self.k(zt).view(B, T, self.h, self.hd).transpose(1, 2)    # (B,h,T,hd)
        v = self.v(zt).view(B, T, self.h, self.hd).transpose(1, 2)
        attn_mask = None
        if text_keep is not None:                                     # (B,T) True = real text token
            attn_mask = text_keep[:, None, None, :]                   # (B,1,1,T) bool
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(B, L, p)
        return self.o(out)


class PBConditioning(nn.Module):
    """
    The privileged-basis conditioning module. Structural (present conditional OR unconditional):
        z      = act(down(x))         # 16-d PB features derived from the residual stream (RoPE-free)
        z_pre  = z                    # exposed to the FiLIP head BEFORE any text is written in
        z     += tanh(g_x) * xattn(z, text)     # text writes into the PB subspace (gated)
        x     += tanh(g_up) * up(z)             # PB subspace written back to residual (zero-init gate)

    The elementwise `act` in 16-d is what makes the basis *privileged* (standard axes are special).
    Superposition can still occur (cf. Toy Models), but features prefer to axis-align, which is what
    makes the FiLIP alignment and any later SAE-style read-out meaningful.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        p = cfg.pb_dim
        self.down = nn.Linear(cfg.d_model, p, bias=False)
        self.act = nn.GELU()                      # try ReLU / SoLU for a harder privileged basis
        self.text_proj = nn.Linear(cfg.text_dim, p, bias=False)   # BiomedBERT -> this layer's PB space
        self.cross = PBCrossAttention(cfg)
        self.up = nn.Linear(p, cfg.d_model, bias=False)
        # Affine-free RMS norms on the writeback paths. They strip magnitude so the model
        # CANNOT route a large conditioning signal through a near-zero gate by inflating up/cross
        # weights; the tanh gate becomes the honest throttle (contribution norm ~= |tanh(g)|*sqrt(dim),
        # i.e. the fractional size of the write relative to the residual). Keeps the gate interpretable.
        self.up_norm = RMSNorm(cfg.d_model, affine=False)
        self.cross_norm = RMSNorm(p, affine=False)
        # Flamingo-style zero-init gates: tanh(0)=0 makes delta==0 at init (identity -> a clean
        # unconditional OADM), while the gate itself still receives gradient (d delta/d g = up(z) != 0),
        # so it opens during training. NB: do NOT also zero up.weight — zeroing both the gate and the
        # projection freezes the whole writeback path at zero (its gradient vanishes). Exactly one.
        self.g_cross = nn.Parameter(torch.zeros(1))
        self.g_up = nn.Parameter(torch.zeros(1))

    def forward(self, x_normed, te, gen_keep):
        # `te` (B, 1+T, text_dim) ALWAYS carries the learned null token at index 0, so the
        # unconditional pass flows through the same cross-attention path (attending to null)
        # rather than skipping it. That is what makes CFG's (cond - uncond) well-calibrated.
        z = self.act(self.down(x_normed))         # (B,L,p) privileged-basis protein features
        z_pre = z                                 # for FiLIP: the protein's *own* representation
        zt = self.text_proj(te)                   # (B, 1+T, p) text (+ null slot) in the PB subspace
        z = z + torch.tanh(self.g_cross) * self.cross_norm(self.cross(z, zt, gen_keep))
        delta = torch.tanh(self.g_up) * self.up_norm(self.up(z))   # (B,L,d)
        return delta, z_pre, zt


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    """Standard pre-norm transformer block (upstream / downstream / non-PB middle)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, keep_mask, *_):
        x = x + self.attn(self.n1(x), keep_mask)
        x = x + self.ff(self.n2(x))
        return x, None, None


class PBBlock(nn.Module):
    """Middle block that additionally hosts a privileged-basis text-conditioning sublayer."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)
        self.npb = RMSNorm(cfg.d_model)
        self.pb = PBConditioning(cfg)

    def forward(self, x, keep_mask, te, gen_keep):
        x = x + self.attn(self.n1(x), keep_mask)
        x = x + self.ff(self.n2(x))
        delta, z_pre, zt = self.pb(self.npb(x), te, gen_keep)
        x = x + delta
        return x, z_pre, zt


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class RecurrentOADM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.up_layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_upstream))
        self.mid_layers = nn.ModuleList(
            PBBlock(cfg) if i in cfg.pb_layers else Block(cfg) for i in range(cfg.n_middle)
        )
        self.down_layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_downstream))

        # gated re-injection of the upstream representation at the start of every recurrence pass.
        # Zero-init: if the recurrent state already carries the partial-sequence context, this stays
        # shut; it opens only if deep unrolling starts to unbalance sequence-context vs. text (which
        # is re-read every pass via the PB cross-attn). Learnable, so it costs nothing if unneeded.
        self.reinject_gate = nn.Parameter(torch.zeros(1))

        # Learned null-condition token (in BiomedBERT input space). Prepended to every example's
        # text sequence; the unconditional / CFG-dropped rows attend to THIS instead of real text.
        # A proper learned null (Ho & Salimans) makes the unconditional distribution a calibrated
        # baseline, so guidance logits = uncond + w*(cond - uncond) actually dial text adherence.
        self.null_text = nn.Parameter(torch.randn(cfg.text_dim) * 0.02)

        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def _build_text(self, text_emb, text_keep, cond_mask, B, device):
        """Prepend the learned null token and build the per-row generative keep mask.

        Returns te (B, 1+T, text_dim), gen_keep (B, 1+T) bool, n_real (int).
          * uncond rows (cond_mask False) attend ONLY the null slot.
          * cond rows attend ONLY their real text tokens (null slot masked off).
        Note: the projection of the real tokens is still computed for every row (masking only
        affects the cross-attention softmax), so FiLIP can use real text even on CFG-dropped rows.
        """
        null = self.null_text.view(1, 1, -1).expand(B, 1, -1)          # (B,1,text_dim)
        if text_emb is None:                                          # fully unconditional
            keep = torch.ones(B, 1, dtype=torch.bool, device=device)
            return null, keep, 0
        T = text_emb.shape[1]
        text_emb = text_emb.to(null.dtype)
        if text_keep is None:
            text_keep = torch.ones(B, T, dtype=torch.bool, device=device)
        if cond_mask is None:
            cond_mask = torch.ones(B, dtype=torch.bool, device=device)
        cm = cond_mask.view(B, 1)
        te = torch.cat([null, text_emb], dim=1)                       # (B, 1+T, text_dim)
        keep = torch.cat([~cm, text_keep & cm], dim=1)                # (B, 1+T)
        return te, keep, T

    def forward(self, tokens, text_emb=None, text_keep=None, cond_mask=None,
                canvas_mask=None, n_recurrence=None, collect_filip="last"):
        """
        tokens:      (B, L) long, includes MASK where unknown; EOS marks length; PAD after EOS
                     is a MODELLED, attended token (predicted as PAD), not attention filler.
        text_emb:    (B, T, text_dim) BiomedBERT token embeddings, or None (fully unconditional).
        text_keep:   (B, T) bool, True = real text token (False = padding).
        cond_mask:   (B,) bool, True = condition this row on its text, False = use the null token
                     (CFG dropout / unlabelled rows). None -> all-True when text_emb is given.
        canvas_mask: (B, L) bool, True = position is part of the modelled canvas (AA/EOS/PAD).
                     None -> full attention (every position modelled). Set it only when the batch
                     has true beyond-canvas filler (e.g. padding past a length bucket's width).
        returns:     logits (B, L, vocab), filip_pairs [(z_pre (B,L,p), zt_real (B,T,p)), ...]
        """
        cfg = self.cfg
        B, L = tokens.shape
        N = n_recurrence or cfg.n_recurrence
        keep_mask = canvas_mask                                 # None -> full (bidirectional) attention

        te, gen_keep, n_real = self._build_text(text_emb, text_keep, cond_mask, B, tokens.device)

        x = self.embed(tokens)
        for l in self.up_layers:
            x, _, _ = l(x, keep_mask)
        h_up = x                                                # upstream output for re-injection

        filip_pairs = []
        for step in range(N):
            x = x + torch.tanh(self.reinject_gate) * h_up       # gated re-injection each pass
            for l in self.mid_layers:
                x, z_pre, zt = l(x, keep_mask, te, gen_keep)
                take = (collect_filip == "all") or (collect_filip == "last" and step == N - 1)
                if take and n_real > 0 and z_pre is not None:
                    filip_pairs.append((z_pre, zt[:, 1:, :]))   # drop null slot -> real text only

        for l in self.down_layers:
            x, _, _ = l(x, keep_mask)

        logits = self.lm_head(self.final_norm(x))
        return logits, filip_pairs


# --------------------------------------------------------------------------------------
# FiLIP-style token-level (residue x word) late interaction
# --------------------------------------------------------------------------------------
def filip_similarity(zp, zt, prot_keep, text_keep, neg=-1e4):
    """
    zp: (B, L, p) protein PB features        prot_keep: (B, L) bool True=real residue
    zt: (B, T, p) text-in-PB features        text_keep: (B, T) bool True=real word
    Returns sim (B, B): rows = proteins, cols = texts; sim[i, j] is the FiLIP score of
    protein i against text j (max-over-tokens, mean-over-query-tokens, both directions).
    """
    zp = F.normalize(zp.float(), dim=-1)
    zt = F.normalize(zt.float(), dim=-1)
    S = torch.einsum("ilp,jtp->ijlt", zp, zt)                   # (B,B,L,T) all cross-sample pairs

    # protein -> text : for each residue, best-matching word; then mean over real residues
    St = S.masked_fill(~text_keep[None, :, None, :], neg)
    p2t = St.max(dim=-1).values                                # (B,B,L)
    pk = prot_keep[:, None, :].float()                         # (B,1,L)
    p2t = (p2t * pk).sum(-1) / pk.sum(-1).clamp(min=1.0)       # (B,B)

    # text -> protein : for each word, best-matching residue; then mean over real words
    Sp = S.masked_fill(~prot_keep[:, None, :, None], neg)
    t2p = Sp.max(dim=-2).values                                # (B,B,T)
    tk = text_keep[None, :, :].float()                         # (1,B,T)
    t2p = (t2p * tk).sum(-1) / tk.sum(-1).clamp(min=1.0)       # (B,B)

    return 0.5 * (p2t + t2p)


def filip_loss(zp, zt, prot_keep, text_keep, temperature=0.07):
    """Symmetric InfoNCE over the (matched) protein/text batch. Apply on the annotated sub-batch."""
    sim = filip_similarity(zp, zt, prot_keep, text_keep) / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    cfg = Config()
    model = RecurrentOADM(cfg).to(device)
    print(f"device={device}  params={count_params(model)/1e6:.1f}M")

    B, L, T = 4, 96, 24
    tokens = torch.randint(0, 20, (B, L), device=device)   # AA ids 0..19
    tokens[:, L // 2:] = cfg.mask_token_id           # a partially-masked OADM canvas
    tokens[:, -9] = cfg.eos_token_id                 # length marker...
    tokens[:, -8:] = cfg.pad_token_id                # ...then modelled PAD (full attention, canvas_mask=None)
    text_emb = torch.randn(B, T, cfg.text_dim, device=device)
    text_keep = torch.ones(B, T, dtype=torch.bool, device=device)
    text_keep[:, T // 2:] = False

    autocast_dev = device if device in ("xpu", "cuda", "cpu") else "cpu"
    with torch.autocast(device_type=autocast_dev, dtype=torch.bfloat16, enabled=(device != "cpu")):
        # conditional
        logits, pairs = model(tokens, text_emb, text_keep, collect_filip="last")
        # unconditional (CFG null branch) — no text pathway engaged
        logits_u, _ = model(tokens, text_emb=None, text_keep=None)

    print("logits:", tuple(logits.shape), "| filip pairs:", len(pairs))
    prot_keep = tokens != cfg.pad_token_id
    zp, zt = pairs[-1]
    loss = filip_loss(zp.float(), zt.float(), prot_keep, text_keep)
    print("FiLIP loss:", loss.detach().item())
    print("CFG delta ||cond-uncond||:", (logits - logits_u).float().norm().detach().item())
