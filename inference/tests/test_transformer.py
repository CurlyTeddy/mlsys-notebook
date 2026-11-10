import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from inference.src.transformer import MLP, GroupQueryAttention


def make_inputs(B, T, S, d_model, k_dim, v_dim, requires_grad=False):
    g = torch.Generator().manual_seed(1)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64, requires_grad=requires_grad)
    return mk(B, T, d_model), mk(B, S, k_dim), mk(B, S, v_dim)

def make_masks(B, T, S):
    g = torch.Generator().manual_seed(2)
    attn_mask = torch.rand(T, S, generator=g) > 0.7
    attn_mask[:, 0] = False
    kpm = torch.zeros(B, S, dtype=torch.bool)
    kpm[1, -3:] = True
    return attn_mask, kpm

def ref_forward(m, q, k, v, key_padding_mask=None, attn_mask=None, is_causal=False):
    B, T, _ = q.shape
    S = k.shape[1]
    n_rep = m.q_heads // m.kv_heads
    split = lambda x, H: x.view(B, -1, H, x.shape[-1] // H).transpose(1, 2)
    Q = split(m.q_proj(q), m.q_heads)
    K = split(m.k_proj(k), m.kv_heads).repeat_interleave(n_rep, dim=1)
    V = split(m.v_proj(v), m.kv_heads).repeat_interleave(n_rep, dim=1)

    if m.rope_parameters:
        D = Q.shape[-1]
        cfg = LlamaConfig(head_dim=D, max_position_embeddings=131072, rope_parameters=m.rope_parameters)
        rot = LlamaRotaryEmbedding(config=cfg)
        pos = torch.arange(T)[None]
        cos, sin = rot(Q, pos)
        Q, _ = apply_rotary_pos_emb(Q, Q, cos, sin)
        pos = torch.arange(S)[None]
        cos, sin = rot(K, pos)
        K, _ = apply_rotary_pos_emb(K, K, cos, sin)

    masked = torch.zeros(B, 1, T, S, dtype=torch.bool)
    if key_padding_mask is not None:
        masked = masked | key_padding_mask[:, None, None, :]
    if attn_mask is not None:
        masked = masked | attn_mask
    if is_causal:
        masked = masked | torch.ones(T, S, dtype=torch.bool).triu(1)
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=~masked)
    return m.o_proj(out.transpose(1, 2).reshape(B, T, -1))

@pytest.mark.parametrize("q_heads,kv_heads", [(8, 8), (8, 4), (8, 2), (8, 1)])
@pytest.mark.parametrize("T,S", [(10, 10), (1, 12), (7, 13)])
@pytest.mark.parametrize("mode", ["none", "causal", "attn_mask", "kpm", "attn_mask+kpm"])
@pytest.mark.parametrize("rope_params", [RopeParameters(
        rope_type="llama3",
        rope_theta=500_000,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ), None])
def test_forward_matches_sdpa(q_heads: int, kv_heads: int, T: int, S: int, mode: str, rope_params: RopeParameters | None):
    if mode == "causal" and T != S:
        pytest.skip("causal alignment differs for T != S")

    d_model, k_dim, v_dim = 64, 48, 56
    model = GroupQueryAttention(q_heads,
                                kv_heads,
                                d_model=d_model,
                                d_k=64,
                                k_dim=k_dim,
                                v_dim=v_dim,
                                layer_index=0,
                                rope_parameters=rope_params).double().eval()
    q, k, v = make_inputs(2, T, S, d_model, k_dim, v_dim)
    attn_mask, kpm = make_masks(2, T, S)
    kw = {
        "is_causal": (mode == "causal"),
        "attn_mask": attn_mask if "attn_mask" in mode else None,
        "key_padding_mask": kpm if "kpm" in mode else None,
    }
    torch.testing.assert_close(model(q, k, v, **kw), ref_forward(model, q, k, v, **kw), rtol=1e-7, atol=1e-7)


def ref_mlp(t: torch.Tensor, d_model: int, d_ff: int, bias: bool, m: MLP):
    cfg = LlamaConfig(hidden_size=d_model, intermediate_size=d_ff, mlp_bias=bias, hidden_act="silu")
    mlp = LlamaMLP(cfg)
    mlp.load_state_dict(m.state_dict(), strict=True)

    return mlp(t)


@pytest.mark.parametrize("B", [1, 10, 100])
@pytest.mark.parametrize("T", [1, 10, 20])
@pytest.mark.parametrize("d_model,d_ff", [(32, 128), (64, 256), (256, 1024)])
@pytest.mark.parametrize("bias", [True, False])
def test_feed_forward(B: int, T: int, d_model: int, d_ff: int, bias: bool):
    g = torch.Generator().manual_seed(1)
    rand_tensor = torch.randn(B, T, d_model, generator=g)
    mlp = MLP(d_model, d_ff, bias=bias)

    torch.testing.assert_close(mlp(rand_tensor), ref_mlp(rand_tensor, d_model, d_ff, bias, mlp), rtol=1e-5, atol=1e-5)