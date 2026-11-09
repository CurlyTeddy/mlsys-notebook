from typing import Optional
from torch import nn

import copy
import math
import torch


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim: int=512, num_heads: int=8, k_dim: int=512, v_dim: int=512):
        # default values are the hyperparameters the paper used
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("Embedding dimension is not divisible to number of heads")
        
        self.num_heads = num_heads

        # parameters from multiple heads are stacked as [W_1, ..., W_h]
        # W_n is of size (embedding_dim, embedding_dim // head)
        self.W_Q = nn.Parameter(torch.empty(size=(embedding_dim, embedding_dim)))
        self.W_K = nn.Parameter(torch.empty(size=(k_dim, embedding_dim)))
        self.W_V = nn.Parameter(torch.empty(size=(v_dim, embedding_dim)))
        self.W_O = nn.Parameter(torch.empty((embedding_dim, embedding_dim)))
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)
    
    def forward(self,
                query: torch.Tensor,   # (batch, target_length, embedding_dim)
                key: torch.Tensor,     # (batch, source_length, k_dim)
                value: torch.Tensor,   # (batch, source_length, v_dim)
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        # the key_padding_mask is used to mask out the paddings in sequences
        # it should have shape (N, S) where N is the size of batches and S is the max sequence length
        # attn_mask is for the masked multi-head attention in the decoder, masking out the leftward relationship of tokens
        batch, target_length, embedding_dim = query.shape
        source_length = key.shape[1]

        if value.shape[1] != source_length:
            raise ValueError("key and value must have the same length")
    
        d_k = embedding_dim // self.num_heads
        Q = (query @ self.W_Q).view(size=(batch, target_length, self.num_heads, d_k)).transpose(1, 2)
        K = (key @ self.W_K).view(size=(batch, source_length, self.num_heads, d_k)).transpose(1, 2)
        V = (value @ self.W_V).view(size=(batch, source_length, self.num_heads, d_k)).transpose(1, 2)

        score = Q @ K.transpose(-1, -2) / math.sqrt(d_k)

        # assuming both are boolean masks
        # True means ignore that position
        if key_padding_mask is not None:
            score.masked_fill_(key_padding_mask[:, torch.newaxis, torch.newaxis, :] == 1, float("-inf"))
        
        if attn_mask is not None:
            score.masked_fill_(attn_mask == 1, float("-inf"))

        weights = torch.softmax(score, dim=-1)
        
        return torch.cat((weights @ V).unbind(dim=1), dim=-1) @ self.W_O, weights
    

class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim=512, dim_feedforward=2048):
        super().__init__()
        self.seq_layer = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, embedding_dim)
        )
    
    def forward(self, X: torch.Tensor):
        return self.seq_layer(X)
    

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int=512, num_heads: int=8, feedforward_dim=2048, dropout: float=0.1):
        super().__init__()
        self.mha = MultiheadAttention(embedding_dim, num_heads, embedding_dim, embedding_dim)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForwardLayer(embedding_dim, feedforward_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None):
        # encoder sometimes uses attn_mask to block token-to-token attention
        # though attn_mask is more often used as a causal mask in decoder
        mha_output = self.dropout(self.mha(source, source, source, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0])
        mha_output = self.norm_1(source + mha_output)
        return self.norm_2(mha_output + self.dropout(self.ff(mha_output)))
    

class Encoder(nn.Module):
    def __init__(self, encoder_layer: EncoderLayer, num_layers: int=6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
    
    def forward(self, source: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None, attn_mask: Optional[torch.Tensor]=None):
        for layer in self.layers:
            source = layer(source, key_padding_mask, attn_mask)
        
        return source
    

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim: int=512, num_heads: int=8, k_dim: int=512, v_dim: int=512, feedforward_dim=2048, dropout: float=0.1):
        super().__init__()
        self.self_mha = MultiheadAttention(embedding_dim, num_heads, embedding_dim, embedding_dim)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.cross_mha = MultiheadAttention(embedding_dim, num_heads, k_dim, v_dim)
        self.norm_2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedForwardLayer(embedding_dim, feedforward_dim)
        self.norm_3 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                target: torch.Tensor,
                memory: torch.Tensor,
                target_key_padding_mask: Optional[torch.Tensor]=None,
                target_mask: Optional[torch.Tensor]=None,
                memory_key_padding_mask: Optional[torch.Tensor]=None,
                memory_mask: Optional[torch.Tensor]=None):
        
        # self attention in decoder also has paddings and need causal mask
        self_mha_output = self.dropout(self.self_mha(target, target, target, key_padding_mask=target_key_padding_mask, attn_mask=target_mask)[0])
        self_mha_output = self.norm_1(target + self_mha_output)

        # cross attention needs the masks from encoder to do the same paddings and token-to-token attention
        cross_mha_ouput = self.dropout(self.cross_mha(self_mha_output, memory, memory, key_padding_mask=memory_key_padding_mask, attn_mask=memory_mask)[0])
        cross_mha_ouput = self.norm_2(self_mha_output + cross_mha_ouput)

        return self.norm_3(cross_mha_ouput + self.dropout(self.ff(cross_mha_ouput)))
    

class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, num_layers: int=6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
    
    def forward(self,
                target: torch.Tensor,
                memory: torch.Tensor,
                target_key_padding_mask: Optional[torch.Tensor]=None,
                target_mask: Optional[torch.Tensor]=None,
                memory_key_padding_mask: Optional[torch.Tensor]=None,
                memory_mask: Optional[torch.Tensor]=None):
        
        for layer in self.layers:
            target = layer(target, memory, target_key_padding_mask, target_mask, memory_key_padding_mask, memory_mask)
        
        return target
    

class Transformer(nn.Module):
    def __init__(self,
                 embedding_dim: int=512,
                 num_heads: int=8,
                 encoder_num_layers: int=6,
                 decoder_num_layers: int=6,
                 feedforward_dim: int=2048,
                 dropout: float=0.1):
        super().__init__()
        self.encoder = Encoder(EncoderLayer(embedding_dim, num_heads, feedforward_dim, dropout), encoder_num_layers)
        self.decoder = Decoder(DecoderLayer(embedding_dim, num_heads, feedforward_dim=feedforward_dim, dropout=dropout), decoder_num_layers)
    
    # in the vanilla transformer source_mask == memory_mask and source_key_padding_mask == memory_key_padding_mask
    # but not always true in other transformer models
    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor,
                source_mask: Optional[torch.Tensor]=None,
                source_key_padding_mask: Optional[torch.Tensor]=None,
                target_mask: Optional[torch.Tensor]=None,
                target_key_padding_mask: Optional[torch.Tensor]=None,
                memory_mask: Optional[torch.Tensor]=None,
                memory_key_padding_mask: Optional[torch.Tensor]=None,
                target_is_causal: int=False):
        
        if target_is_causal:
            target_length = target.shape[1]
            target_mask = torch.triu(
                torch.ones(target_length, target_length, dtype=torch.bool),
                diagonal=1
            ).to(source.device)

        memory = self.encoder(source, attn_mask=source_mask, key_padding_mask=source_key_padding_mask)
        return self.decoder(target, memory, target_key_padding_mask=target_key_padding_mask, target_mask=target_mask, memory_key_padding_mask=memory_key_padding_mask, memory_mask=memory_mask)
    
