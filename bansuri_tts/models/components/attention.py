from typing import Optional, Tuple

import math
import torch
from torch import nn

def build_rope_cache(
    seq_len: int, n_elem: int, device: Optional[torch.device] = None, base: int = 10000, condense_ratio: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)

class SelfAttention(nn.Module):
    def __init__(self, n_head, d_head, n_query_groups, d_embed, bias=False):
        super(SelfAttention, self).__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.n_query_groups = n_query_groups
        self.d_embed = d_embed

        shape = (n_head + 2 * n_query_groups) * d_head
        self.qkv_proj = nn.Linear(d_embed, shape, bias=bias)
        self.out_proj = nn.Linear(n_head * d_head, d_embed, bias=bias)

    def forward(self, 
            x: torch.Tensor, 
            cos: torch.Tensor, 
            sin: torch.Tensor, 
            mask: torch.Tensor, 
            # input_pos
        ) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.qkv_proj(x)

        # adaptive logic for MQA, GQA. Falls back to MQA.
        q_per_kv = self.n_head // self.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(B, T, self.n_query_groups, total_qkv, self.d_head)
        qkv = qkv.permute(0, 2, 3, 1, 4)

        q, k, v = qkv.split((q_per_kv, 1, 1), dim=2)

        if self.config.n_query_groups != self.config.n_head and self.config.n_query_groups != 1:
            k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
            v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B, -1, T, self.config.head_size)  # (B, nh_q, T, hs)
        k = k.reshape(B, -1, T, self.config.head_size)  # (B, nh_k, T, hs)
        v = v.reshape(B, -1, T, self.config.head_size)  # (B, nh_v, T, hs)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        y = self.scaled_dot_product_attention(q, k, v, mask)

        y = y.reshape(B, T, self.config.head_size * self.config.n_head)  # re-assemble all head outputs side by side

        # output projection
        return self.out_proj(y)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.head_size)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale
        )
        return y.transpose(1, 2)
