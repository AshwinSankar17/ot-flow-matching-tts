from typing import Optional
from functools import partial

import math
import torch
from torch import nn

from bansuri_tts.models.components.attention import SelfAttention, build_rope_cache
from bansuri_tts.models.components.mlp import MLP

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

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
        # print(t.shape)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # print(embedding.shape)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiTBlock(nn.Module):
    def __init__(self, n_head, d_head, d_embed, n_query_groups, rope_n_elem, ff_dim, bias=False):
        super(DiTBlock, self).__init__()
        n_query_groups = n_head if n_query_groups is None else n_query_groups
        self.norm_1 = nn.LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(n_head, d_head, n_query_groups, d_embed, rope_n_elem, bias=bias)
        self.norm_2 = nn.LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)
        approx_gelu = partial(nn.GELU, approximate="tanh")
        self.mlp = MLP(d_embed, ff_dim, d_embed, approx_gelu)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_embed, 6 * d_embed, bias=True)
        )

    def forward(self, x, condition, cos, sin, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm_1(x), shift_msa, scale_msa), cos, sin, mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm_2(x), shift_mlp, scale_mlp), mask)
        return x

class PixartDiTBlock(nn.Module):
    def __init__(self, n_head, d_head, d_embed, n_query_groups, rope_n_elem, ff_dim, bias=False):
        super(PixartDiTBlock, self).__init__()
        n_query_groups = n_head if n_query_groups is None else n_query_groups
        self.norm_1 = nn.LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(n_head, d_head, n_query_groups, d_embed, rope_n_elem, bias=bias)
        self.cross_attn = nn.MultiheadAttention(d_embed, n_head, batch_first=True, bias=bias)
        self.norm_2 = nn.LayerNorm(d_embed, elementwise_affine=False, eps=1e-6)
        approx_gelu = partial(nn.GELU, approximate="tanh")
        self.mlp = MLP(d_embed, ff_dim, d_embed, approx_gelu)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(d_embed, 6 * d_embed, bias=True)
        # )

    def forward(self, x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, condition, cos, sin, mask):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
        # print(x.shape, condition.shape)
        # exit()
        x = x + gate_msa * self.attn(modulate(self.norm_1(x), shift_msa, scale_msa), cos, sin, mask)
        x = x + self.cross_attn(x, condition, condition)[0]
        x = x + gate_mlp * self.mlp(modulate(self.norm_2(x), shift_mlp, scale_mlp), mask)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Conv1d(hidden_size, out_channels, kernel_size=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x.transpose(1,2))
        return x.transpose(1,2)

class DiT(nn.Module):
    def __init__(
        self, 
        n_head=8, 
        d_head=128, 
        d_embed=512, 
        n_layers=6,
        n_mels=100,
        ff_dim=2048,
        rope_base=10000,
        rope_condense_ratio=1,
        rotary_percentage=1.0,
        n_query_groups=None, 
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = d_embed
        self.out_channels = n_mels * 2 if self.learn_sigma else n_mels
        self.rope_base = rope_base
        self.rotary_percentage = rotary_percentage
        self.rope_condense_ratio = rope_condense_ratio
        self.rope_n_elem = int(self.rotary_percentage * d_head)

        self.rope_cache = None
        self.max_seq_length = 4096
        self.t_embedder = TimestepEmbedder(d_embed)

        self.noise_adapter = nn.Linear(n_mels, d_embed)
        self.condn_adapter = nn.Linear(n_mels, d_embed)

        self.blocks = nn.ModuleList([
            DiTBlock(n_head, d_head, d_embed, n_query_groups, self.rope_n_elem, ff_dim) for _ in range(n_layers)
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
        nn.init.constant_(self.post.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.post.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.post.linear.weight, 0)
        nn.init.constant_(self.post.linear.bias, 0)

    def build_rope_cache(self, device: Optional[torch.device]=None) -> torch.Tensor:
        self.rope_cache = build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.rope_n_elem,
            device=device,
            condense_ratio=self.rope_condense_ratio,
            base=self.rope_base
        )
    
    def forward(self, x, t, y, mask):
        y = self.condn_adapter(y)
        x = self.noise_adapter(x)
        t = self.t_embedder(t).unsqueeze(1)
        # print(x.shape, y.shape, t.shape)
        c = t + y                              # (N, T, D)
        T = x.size(1)
        if self.max_seq_length < T:
            self.max_seq_length = T
            self.build_rope_cache(x.device)
        if self.rope_cache is None:
            self.build_rope_cache(x.device)
        
        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]

        for block in self.blocks:
            x = block(x, c, cos, sin, mask)      # (N, T, D)
        x = self.post(x, c)
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


class PixartDiT(nn.Module):
    def __init__(
        self, 
        n_head=16, 
        d_head=64, 
        d_embed=512, 
        n_layers=6,
        n_mels=100,
        ff_dim=2048,
        rope_base=10000,
        rope_condense_ratio=1,
        rotary_percentage=1.0,
        n_query_groups=None, 
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = d_embed
        self.out_channels = n_mels * 2 if self.learn_sigma else n_mels
        self.rope_base = rope_base
        self.rotary_percentage = rotary_percentage
        self.rope_condense_ratio = rope_condense_ratio
        self.rope_n_elem = int(self.rotary_percentage * d_head)

        self.rope_cache = None
        self.max_seq_length = 4096
        self.t_embedder = TimestepEmbedder(d_embed)

        self.noise_adapter = nn.Linear(n_mels, d_embed)
        self.condn_adapter = nn.Linear(n_mels, d_embed)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_embed, 6 * d_embed, bias=True)
        )

        self.blocks = nn.ModuleList([
            PixartDiTBlock(n_head, d_head, d_embed, n_query_groups, self.rope_n_elem, ff_dim) for _ in range(n_layers)
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
        # for block in self.blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.post.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.post.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.post.linear.weight, 0)
        nn.init.constant_(self.post.linear.bias, 0)

    def build_rope_cache(self, device: Optional[torch.device]=None) -> torch.Tensor:
        self.rope_cache = build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.rope_n_elem,
            device=device,
            condense_ratio=self.rope_condense_ratio,
            base=self.rope_base
        )
    
    def forward(self, x, t, y, mask):
        y = self.condn_adapter(y)
        x = self.noise_adapter(x)
        t = self.t_embedder(t).unsqueeze(1)
        # print(x.shape, y.shape, t.shape)
        # c = t + y                              # (N, T, D)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)

        T = x.size(1)
        if self.max_seq_length < T:
            self.max_seq_length = T
            self.build_rope_cache(x.device)
        if self.rope_cache is None:
            self.build_rope_cache(x.device)
        
        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]

        for block in self.blocks:
            x = block(x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, y, cos, sin, mask)      # (N, T, D)
        x = self.post(x, t)
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