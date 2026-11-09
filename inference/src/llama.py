import json
from pathlib import Path

import bitsandbytes as bnb
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformer import *
from transformers import AutoTokenizer, GenerationConfig, LlamaConfig


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=config.dtype)
        self.layers = nn.ModuleList(
            DecoderLayer(
                q_heads=config.num_attention_heads,
                kv_heads=config.num_key_value_heads,
                d_model=config.hidden_size,
                d_k=config.hidden_size // config.num_attention_heads,
                k_dim=config.hidden_size,
                v_dim=config.hidden_size,
                d_ff=config.intermediate_size,
                rope_parameters=config.rope_parameters,
                attn_bias=config.attention_bias,
                mlp_bias=config.mlp_bias,
                eps=config.rms_norm_eps,
                dtype=config.dtype
            ) for _ in range(config.num_hidden_layers)
        )

    def forward(self, input_ids: torch.Tensor, key_padding_mask: torch.Tensor | None=None, attn_mask: torch.Tensor | None=None, is_causal: bool=True):
        embeddings = self.embed_tokens(input_ids)
        for layer in self.layers:
            embeddings = layer(embeddings, key_padding_mask, attn_mask, is_causal)
        return self.norm(embeddings)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)
        self.cfg = config

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor, is_causal: bool):
        return self.lm_head(self.model(input_ids, key_padding_mask=padding_mask, is_causal=is_causal))[:, -1, :]

    def generate(self, gen_config: GenerationConfig, input_ids: torch.Tensor, padding_mask: torch.Tensor, is_causal: bool=True):
        max_length = self.cfg.max_position_embeddings
        batch, length = input_ids.shape
        if length > max_length:
            raise ValueError("Input tokens are longer than max input length.")

        if gen_config.temperature <= 0:
            raise ValueError("Temperature must be larger than 0.")

        device = input_ids.device
        if isinstance(gen_config.eos_token_id, int):
            eos_id = torch.tensor([gen_config.eos_token_id], device=device)
        else:
            eos_id = torch.tensor(gen_config.eos_token_id, device=device)

        self.eval()
        with torch.inference_mode():
            padding_mask = torch.cat([padding_mask, torch.zeros(batch, max_length - length, dtype=torch.bool, device=input_ids.device)], dim=-1)
            while not torch.all(torch.isin(input_ids[:, -1], eos_id)) and input_ids.shape[-1] < max_length:
                logits: torch.Tensor = self.forward(input_ids, padding_mask[:, :input_ids.shape[-1]], is_causal).float() / gen_config.temperature

                # we want to find the position to mask out in top p
                # sort probs, accumulate it, and mask out low p
                sorted_p, sorted_indexes = torch.softmax(logits, dim=-1).sort(descending=True)
                mask = sorted_p.cumsum(dim=-1) <= gen_config.top_p
                mask[..., 1:] = mask[..., :-1].clone()
                 # handle the edge case, when the first p is larger than top_p
                mask[..., 0] = True
                sorted_p *= mask

                # redistribute prob and put prob back to corresponding position
                sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
                vocab_p = torch.zeros_like(sorted_p, device=device).scatter_(dim=-1, index=sorted_indexes, src=sorted_p)
                has_eos = torch.isin(input_ids[:, -1], eos_id)
                next_token_id = torch.where(has_eos.unsqueeze_(dim=-1), eos_id[0], torch.multinomial(vocab_p, num_samples=1))
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                print(f"length: {input_ids.shape[1]}")

        return input_ids


# download weights and configs
path = Path(__file__).parent / "llama31-8b"
if not path.exists():
    snapshot_download(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        local_dir=path,
    )

model_id = "meta-llama/Llama-3.1-8B-Instruct"
cfg = LlamaConfig.from_pretrained(model_id)
gen_cfg = GenerationConfig.from_pretrained(model_id)

with torch.device("meta"):
    model = LlamaForCausalLM(cfg)

def swap_linears(module, skip=("lm_head",)):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name not in skip:
            setattr(module, name, bnb.nn.Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type="nf4", compress_statistics=True,
                device="meta"))
        else:
            swap_linears(child, skip)

swap_linears(model)

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(path / "model.safetensors.index.json") as f:
    indexes = json.load(f)

for name, shard in sorted(indexes["weight_map"].items(), key=lambda x: x[1]):
    mod_name, param_name = name.rsplit('.', 1)
    mod = model.get_submodule(mod_name)

    # weights are only quantized when moving from cpu to gpu
    with safe_open(f"{path}/{shard}", "pt", "cpu") as f:
        if isinstance(mod, bnb.nn.Linear4bit) and param_name == "weight":
            mod.weight = bnb.nn.Params4bit(f.get_tensor(name), quant_type="nf4", compress_statistics=True).to(device)
        else:
            setattr(mod, param_name, nn.Parameter(f.get_tensor(name).to(device)))

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    pad_token="<|finetune_right_pad_id|>",
    padding_side="left",
    model_max_length=4096
)

conversations = [
    [{"role": "user", "content": "What is RoPE?"}],
    [{"role": "system", "content": "Be brief."},
     {"role": "user", "content": "Explain KV caching."}],
]

batch = tokenizer.apply_chat_template(
    conversations,
    add_generation_prompt=True,
    padding=True,
    truncation=True,
    max_length=4096,
    return_tensors="pt",
    return_dict=True
).to(device)

output_ids = model.generate(gen_cfg, batch.input_ids, padding_mask=~batch.attention_mask.bool())
print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))