from torch import nn
from src.transformer import MultiheadAttention

import torch
import torch.testing as tt


def test_self_attention_without_masks():
    target_dim = 512
    base = nn.MultiheadAttention(embed_dim=target_dim, num_heads=8, bias=False, batch_first=True)
    custom = MultiheadAttention()
    with torch.no_grad():
        q_weight, k_weight, v_weight = base.in_proj_weight.chunk(3, dim=0)
        custom.W_Q.copy_(q_weight.T)
        custom.W_K.copy_(k_weight.T)
        custom.W_V.copy_(v_weight.T)
        custom.W_O.copy_(base.out_proj.weight.T)

    seq_length = 10
    x = torch.rand((1, seq_length, target_dim))
    actual, actual_weights = custom(x, x, x)
    expected, expected_weights = base(x, x, x, need_weights=True, average_attn_weights=False)
    tt.assert_close(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
    tt.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-5, equal_nan=True)


def test_self_attention_with_masks():
    target_dim = 512
    base = nn.MultiheadAttention(embed_dim=target_dim, num_heads=8, bias=False, batch_first=True)
    custom = MultiheadAttention()
    with torch.no_grad():
        q_weight, k_weight, v_weight = base.in_proj_weight.chunk(3, dim=0)
        custom.W_Q.copy_(q_weight.T)
        custom.W_K.copy_(k_weight.T)
        custom.W_V.copy_(v_weight.T)
        custom.W_O.copy_(base.out_proj.weight.T)

    seq_length = 10
    x = torch.rand((1, seq_length, target_dim))
    key_padding_mask = torch.rand((1, seq_length)) > 0.5
    attn_mask = torch.rand((seq_length, seq_length)) > 0.5
    actual, actual_weights = custom(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    expected, expected_weights = base(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)

    tt.assert_close(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
    tt.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-5, equal_nan=True)


def test_cross_attention_without_masks():
    target_dim = 512
    key_dim = 256
    value_dim = 128
    base = nn.MultiheadAttention(embed_dim=target_dim, num_heads=8, bias=False, batch_first=True, kdim=key_dim, vdim=value_dim)
    custom = MultiheadAttention(k_dim=key_dim, v_dim=value_dim)
    with torch.no_grad():
        custom.W_Q.copy_(base.q_proj_weight.T)
        custom.W_K.copy_(base.k_proj_weight.T)
        custom.W_V.copy_(base.v_proj_weight.T)
        custom.W_O.copy_(base.out_proj.weight.T)

    target_length = 10
    source_length = 20
    query = torch.rand((1, target_length, target_dim))
    key = torch.rand((1, source_length, key_dim))
    value = torch.rand((1, source_length, value_dim))
    actual, actual_weights = custom(query, key, value)
    expected, expected_weights = base(query, key, value, need_weights=True, average_attn_weights=False)
    
    tt.assert_close(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
    tt.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-5, equal_nan=True)

def test_cross_attention_with_masks():
    target_dim = 512
    key_dim = 256
    value_dim = 128
    base = nn.MultiheadAttention(embed_dim=target_dim, num_heads=8, bias=False, batch_first=True, kdim=key_dim, vdim=value_dim)
    custom = MultiheadAttention(k_dim=key_dim, v_dim=value_dim)
    with torch.no_grad():
        custom.W_Q.copy_(base.q_proj_weight.T)
        custom.W_K.copy_(base.k_proj_weight.T)
        custom.W_V.copy_(base.v_proj_weight.T)
        custom.W_O.copy_(base.out_proj.weight.T)

    target_length = 10
    source_length = 20
    query = torch.rand((1, target_length, target_dim))
    key = torch.rand((1, source_length, key_dim))
    value = torch.rand((1, source_length, value_dim))

    key_padding_mask = torch.rand((1, source_length)) > 0.5
    attn_mask = torch.rand((target_length, source_length)) > 0.5
    actual, actual_weights = custom(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    expected, expected_weights = base(query, key, value, need_weights=True, average_attn_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

    tt.assert_close(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
    tt.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-5, equal_nan=True)

def test_batch_cross_attention_with_masks():
    target_dim = 512
    key_dim = 256
    value_dim = 128
    base = nn.MultiheadAttention(embed_dim=target_dim, num_heads=8, bias=False, batch_first=True, kdim=key_dim, vdim=value_dim)
    custom = MultiheadAttention(k_dim=key_dim, v_dim=value_dim)
    with torch.no_grad():
        custom.W_Q.copy_(base.q_proj_weight.T)
        custom.W_K.copy_(base.k_proj_weight.T)
        custom.W_V.copy_(base.v_proj_weight.T)
        custom.W_O.copy_(base.out_proj.weight.T)

    target_length = 10
    source_length = 20
    batch_size = 2
    query = torch.rand((batch_size, target_length, target_dim))
    key = torch.rand((batch_size, source_length, key_dim))
    value = torch.rand((batch_size, source_length, value_dim))

    key_padding_mask = torch.rand((batch_size, source_length)) > 0.5
    attn_mask = torch.rand((target_length, source_length)) > 0.5
    actual, actual_weights = custom(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    expected, expected_weights = base(query, key, value, need_weights=True, average_attn_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

    tt.assert_close(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=True)
    tt.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-5, equal_nan=True)