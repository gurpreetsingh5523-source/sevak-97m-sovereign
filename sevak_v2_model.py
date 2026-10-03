#!/usr/bin/env python3
"""
Sevak-22M v2 — Modern Architecture
RMSNorm + RoPE + SwiGLU + QK-Norm + Tied Embeddings
8 layers, 512 dim, 16K vocab, 1024 context

"ਉੱਤਮ ਮਤਿ ਰਿਦੈ ਤੁਮ੍ਹ ਆਓ"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============ CONFIG ============
D = 768          # model dim (was 512)
NH = 12          # attention heads (was 8)
NL = 12          # layers (was 8)
V = 16000        # vocab
C = 1024         # context
FF = 2048        # FFN dim (~2.7x D, SwiGLU)

# ============ RMSNorm ============
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

# ============ RoPE ============
class RoPE(nn.Module):
    def __init__(self, dim, max_len=1024, base=10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])
    def forward(self, x, seq_len):
        # x: (B, NH, T, HD)
        if seq_len > self.max_len:
            self._build_cache(seq_len)
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        # Split and rotate
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        x_rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rot * sin

# ============ SwiGLU ============
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ============ Attention with QK-Norm ============
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = NH
        self.hd = D // NH
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.rope = RoPE(self.hd, C)
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        # RoPE
        q = self.rope(q, T)
        k = self.rope(k, T)
        # Attention
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, D)
        return self.o(a)

# ============ Block ============
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(D)
        self.attn = Attention()
        self.norm2 = RMSNorm(D)
        self.ffn = SwiGLU(D, FF)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# ============ Sevak-22M v2 ============
class SevakV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(NL)])
        self.norm = RMSNorm(D)
        # Tied embeddings — head shares weight with tok
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.tok.weight  # TIED
        # v1 ਸਬਕ: default init (N(0,1)) ਨਾਲ logits explode → CE 668
        # 0.02 std ਨਾਲ CE ≈ ln(V) ≈ 9.7 ਤੋਂ ਸ਼ੁਰੂ
        self.apply(self._init_weights)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        # Residual output scaling: 0.02/sqrt(2*NL) — deep net stability
        resid_std = 0.02 / math.sqrt(2 * NL)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, mean=0.0, std=resid_std)
            nn.init.normal_(b.ffn.w2.weight, mean=0.0, std=resid_std)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, ids):
        x = self.tok(ids)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return self.head(x)
    def layer_feats(self, ids, upto=4):
        x = self.tok(ids)
        for b in self.blocks[:upto]:
            x = b(x)
        return x
    def generate(self, ids, max_len=50, temp=0.7, top_p=0.9):
        for _ in range(max_len):
            if ids.shape[1] >= C:
                break
            with torch.no_grad():
                logits = self(ids[:, -C:])
            next_logits = logits[0, -1, :] / temp
            # Top-p sampling
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_idx_to_remove = cum_probs > top_p
            sorted_idx_to_remove[1:] = sorted_idx_to_remove[:-1].clone()
            sorted_idx_to_remove[0] = 0
            idx_to_remove = sorted_idx[sorted_idx_to_remove]
            next_logits[idx_to_remove] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
            if next_id.item() == 0:  # EOS
                break
        return ids

# ============ PARAMETER COUNT ============
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    m = SevakV2()
    n = count_params(m)
    print(f"Sevak-100M v2: {n:,} parameters ({n/1e6:.1f}M)")
    print(f"  Config: {NL} layers, {D} dim, {NH} heads, {V} vocab, {C} context")
    print(f"  FFN: {FF} (SwiGLU), Tied embeddings: Yes")
    # Test forward
    x = torch.randint(0, V, (1, 10))
    out = m(x)
    print(f"  Forward test: input {x.shape} → output {out.shape}")
    print("  ✓ Model ready!")
