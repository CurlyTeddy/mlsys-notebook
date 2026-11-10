from typing import Literal

import torch
from torch import nn
from transformers.modeling_rope_utils import RopeParameters

from inference.src.cache import KVCacheHandle


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, dtype: torch.dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        return x / (((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5) * self.weight

class GroupQueryAttention(nn.Module):
    """Grouped-query attention (GQA).

    Queries use ``q_heads`` heads, while keys and values share a smaller set of
    ``kv_heads`` heads. Each key/value head serves a group of
    ``q_heads // kv_heads`` query heads. Setting ``kv_heads == q_heads`` gives
    standard multi-head attention, and ``kv_heads == 1`` gives multi-query
    attention.

    Args:
        q_heads (int): Number of query heads. Must be divisible by ``kv_heads``.
        kv_heads (int): Number of key/value heads, shared across groups of
            query heads.
        d_model (int): Model embedding dimension. This is the feature size of
            the query input and of the block's output.
        d_k (int): Dimension of each attention head, used for queries, keys
            and values. Attention scores are scaled by ``1 / sqrt(d_k)``.
        k_dim (int): Feature dimension of the key input. Equals ``d_model`` for
            self-attention and may differ for cross-attention.
        v_dim (int): Feature dimension of the value input. Equals ``d_model``
            for self-attention and may differ for cross-attention.
        rope_parameters (RopeParameters | None, optional): Configuration for
            rotary position embeddings applied to queries and keys. If
            ``None``, no rotary embedding is applied. Defaults to ``None``.
        attn_bias (bool, optional): Whether the query, key, value and output
            projections include a bias term. Defaults to ``True``.
        dtype (torch.dtype): The data type of linear parameters.
    """
    def __init__(
            self,
            q_heads: int,
            kv_heads: int,
            d_model: int,
            d_k: int,
            k_dim: int,
            v_dim: int,
            layer_index: int,
            rope_parameters: RopeParameters | None = None,
            attn_bias=True,
            dtype: torch.dtype=torch.float32
    ):
        if q_heads % kv_heads != 0:
            raise ValueError("q_heads is not divisible by kv_heads.")

        if d_model % q_heads != 0:
            raise ValueError("d_model is not divisible by q_heads")

        super().__init__()
        group_size = q_heads // kv_heads
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.rope_parameters = rope_parameters
        self.layer_index = layer_index

        self.q_proj = nn.Linear(d_model, d_k * q_heads, attn_bias, dtype=dtype)
        self.k_proj = nn.Linear(k_dim, d_k * kv_heads, attn_bias, dtype=dtype)
        self.v_proj = nn.Linear(v_dim, d_model // group_size, attn_bias, dtype=dtype)
        self.o_proj = nn.Linear(d_model, d_model, attn_bias, dtype=dtype)

    def _scale_rope(self, freq: torch.Tensor) -> torch.Tensor:
        original_max_pos = self.rope_parameters["original_max_position_embeddings"]
        factor = self.rope_parameters["factor"]
        high_freq_factor, low_freq_factor = self.rope_parameters["high_freq_factor"], self.rope_parameters["low_freq_factor"]

        ratio = original_max_pos / (2 * torch.pi / freq)
        smooth_factor = ((ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)).clamp(0, 1)

        return smooth_factor * freq + (1 - smooth_factor) * freq / factor


    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,   # mask paddings in key
            attn_mask: torch.Tensor | None = None,   # usually for causal mask
            is_causal=False,
            cache: KVCacheHandle | None=None
    ):
        if attn_mask is not None and is_causal:
            raise ValueError("Cannot specify attn_mask when is_causal is True.")

        if q.shape[-1] != self.q_proj.in_features or k.shape[-1] != self.k_proj.in_features or v.shape[-1] != self.v_proj.in_features:
            raise ValueError("Input dimension does not match projection dimension.")

        # projection
        batch, t, _ = q.shape
        s = k.shape[1]
        group = self.q_heads // self.kv_heads
        head_dim = self.k_proj.out_features // self.kv_heads
        Q = self.q_proj(q).view(batch, t, self.kv_heads, group, head_dim)
        K = self.k_proj(k).view(batch, s, self.kv_heads, head_dim)
        v_head_dim = self.v_proj.out_features // self.kv_heads
        V = self.v_proj(v).view(batch, s, self.kv_heads, v_head_dim)

        # add rope positional encoding
        # output dimension of pretrained weights on HuggingFace are interleved as [0, 2, 4, 6...1, 3, 5, 7]
        # whether m and n start from 0 or 1 doesn't matter since RoPE only retains the relative position information
        if self.rope_parameters:
            freq = self.rope_parameters["rope_theta"] ** (-torch.arange(head_dim // 2).repeat(2) * 2 / head_dim)
            freq = self._scale_rope(freq).to(dtype=Q.dtype)
            angle = (freq[None, None, None, ...].repeat(t, 1, 1, 1) * torch.arange(t)[..., None, None, None]).to(device=Q.device)   # t, 1, 1, head_dim
            Q = Q * torch.cos(angle) + torch.cat([-Q[..., head_dim // 2:], Q[..., :head_dim // 2]], dim=-1) * torch.sin(angle)
            angle = (freq[None, None, ...].repeat(s, 1, 1) * torch.arange(s)[..., None, None]).to(device=Q.device)   # s, 1, head_dim
            K = K * torch.cos(angle) + torch.cat([-K[..., head_dim // 2:], K[..., :head_dim // 2]], dim=-1) * torch.sin(angle)

        if cache:
            cache.set(self.layer_index, K, V)
            K, V = cache.get(self.layer_index)

        Q = Q.permute(0, 2, 3, 1, 4)   # batch, k_heads, group, t, head_dim
        K = K.permute(0, 2, 3, 1).unsqueeze(2).repeat_interleave(group, dim=2)   # batch, kv_heads, group, head_dim, s
        score =(Q @ K).view(batch, -1, t, s)   # batch, q_heads, t, s

        # apply masks
        if key_padding_mask is not None:
            score.masked_fill_(key_padding_mask[:, torch.newaxis, torch.newaxis, :], float("-inf"))

        if attn_mask is not None:
            score.masked_fill_(attn_mask, float("-inf"))

        if is_causal:
            score.masked_fill_(torch.triu(torch.ones(t, s, dtype=torch.bool, device=Q.device), diagonal=1), float("-inf"))
        # if input is left padded and the first token in a sequence is a pad, all attention of the first row is float("-inf")
        # 0/0 in softmax produces NaN
        score = torch.softmax(score / head_dim ** 0.5, dim=-1).nan_to_num(0.)   # batch, q_heads, t, s

        V = V.repeat_interleave(group, dim=2).transpose(1, 2)  # batch, q_heads, s, v_dim // (group * k_heads)
        # score @ V has shape batch, q_heads, t, v_dim // (group * k_heads)
        return self.o_proj((score @ V).transpose(1, 2).reshape(batch, t, -1))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, mode: Literal["relu", "silu"]="silu", bias=True, dtype: torch.dtype=torch.float32):
        super().__init__()
        self.mode = mode
        self.up_proj = nn.Linear(d_model, d_ff, bias, dtype=dtype)
        self.down_proj = nn.Linear(d_ff, d_model, bias, dtype=dtype)

        if mode == "silu":
            self.gate_proj = nn.Linear(d_model, d_ff, bias, dtype=dtype)

    def forward(self, x: torch.Tensor):
        if self.gate_proj:
            gated_x = self.gate_proj(x)
            return self.down_proj(self.up_proj(x) * gated_x / (1 + torch.e ** -gated_x))

        if self.mode == "relu":
            return self.down_proj(torch.maximum(0, self.up_proj(x)))

        raise NotImplementedError("Only support silu and relu.")


class DecoderLayer(nn.Module):
    def __init__(
        self,
        q_heads: int,
        kv_heads: int,
        d_model: int,
        d_k: int,
        k_dim: int,
        v_dim: int,
        d_ff: int,
        layer_index: int,
        rope_parameters: RopeParameters | None = None,
        attn_bias=True,
        mlp_bias=True,
        eps: float=1e-5,
        dtype:torch.dtype=torch.float32
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(d_model, eps, dtype)
        self.self_attn = GroupQueryAttention(q_heads, kv_heads, d_model, d_k, k_dim, v_dim, layer_index, rope_parameters, attn_bias, dtype)
        self.mlp = MLP(d_model, d_ff, bias=mlp_bias, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(d_model, eps, dtype)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor, is_casual: bool, cache: KVCacheHandle | None=None):
        norm_x = self.input_layernorm(x)
        attn_out = x + self.self_attn(norm_x, norm_x, norm_x, key_padding_mask, attn_mask, is_casual, cache)
        return attn_out + self.mlp(self.post_attention_layernorm(attn_out))