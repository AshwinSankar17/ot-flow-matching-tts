from typing import Optional
from functools import partial

import math
import torch
from torch import nn

from bansuri_tts.models.components.attention import SelfAttention, build_rope_cache
from bansuri_tts.models.components.mlp import MLP

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    def __init__(self, n_head, d_head, d_embed, n_query_groups, hidden_size, bias=False):
        super(DiTBlock, self).__init__()
        n_query_groups = n_head if n_query_groups is None else n_query_groups
        self.norm_1 = nn.LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(n_head, d_head, n_query_groups, d_embed, bias=bias)
        self.norm_2 = nn.LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)
        approx_gelu = partial(nn.GELU, approximate="tanh")
        self.mlp = MLP(d_embed, d_embed, d_embed, approx_gelu)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, condition, cos, sin, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cos, sin, mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), mask)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Conv1d(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    def __init__(
        self, 
        n_head, 
        d_head, 
        d_embed, 
        n_layers, 
        rope_base=10000,
        rope_condense_ratio=1,
        rotary_percentage=1.0,
        n_query_groups=None, 
        learn_sigma=False,
    ):
        self.learn_sigma = learn_sigma
        self.in_channels = d_embed
        self.out_channels = d_embed * 2 if self.learn_sigma else d_embed
        self.rope_base = rope_base
        self.rotary_percentage = rotary_percentage
        self.rope_condense_ratio = rope_condense_ratio
        self.rope_n_elem = int(self.rotary_percentage * n_head)

        self.rope_cache = None
        self.max_seq_length = None
        self.t_embedder = TimestepEmbedder(d_embed)

        self.blocks = nn.ModuleList([
            DiTBlock(n_head, d_head, d_embed, n_query_groups) for _ in range(n_layers)
        ])
        self.post = FinalLayer(d_embed, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def build_rope_cache(self, device: Optional[torch.device]=None) -> torch.Tensor:
        self.rope_cache = build_rope_cache(
            seq_length=self.max_seq_length,
            n_elem=self.rope_n_elem,
            device=device,
            condense_ratio=self.rope_condense_ratio,
            base=self.rope_base
        )
    
    def forward(self, x, t, y, mask):
        t = self.t_embedder(t)
        c = t + y                                # (N, T, D)
        if self.max_seq_length < x.size(1):
            self.max_seq_length = x.size(1)
            self.build_rope_cache(x.device)
        cos, sin = self.rope_cache
        for block in self.blocks:
            x = block(x, c, cos, sin, mask)      # (N, T, D)
        x = self.final_layer(x, c)
        return x
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/facebookresearch/DiT/blob/main/models.py#L250
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)