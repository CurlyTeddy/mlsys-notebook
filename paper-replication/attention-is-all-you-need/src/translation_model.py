import shutil

from datasets import load_dataset
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.text import BLEUScore
from transformer import Transformer
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from typing import Optional


import os
import torch


class LanguageModel(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int=512,
                 num_heads: int=8,
                 encoder_num_layers: int=6,
                 decoder_num_layers: int=6,
                 feedforward_dim: int=2048,
                 max_sequence_length: int=512,
                 dropout: float=0.1):
        super().__init__()
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_dim)
        self.register_buffer("positional_encoding", torch.linspace(0, max_sequence_length - 1, max_sequence_length).unsqueeze(dim=1) / torch.pow(1e4, torch.linspace(0, embedding_dim - 1, embedding_dim) // 2 * 2 / embedding_dim))
        torch.sin(self.positional_encoding[:, ::2], out=self.positional_encoding[:, ::2]) # pyright: ignore[reportIndexIssue]
        torch.cos(self.positional_encoding[:, 1::2], out=self.positional_encoding[:, 1::2]) # pyright: ignore[reportIndexIssue]
        self.transformer = Transformer(embedding_dim, num_heads, encoder_num_layers, decoder_num_layers, feedforward_dim, dropout)
        self.head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self,
                source_ids: torch.Tensor,
                target_ids: torch.Tensor,
                source_mask: Optional[torch.Tensor]=None,
                source_key_padding_mask: Optional[torch.Tensor]=None,
                target_mask: Optional[torch.Tensor]=None,
                target_key_padding_mask: Optional[torch.Tensor]=None,
                memory_mask: Optional[torch.Tensor]=None,
                memory_key_padding_mask: Optional[torch.Tensor]=None,
                target_is_causal: bool=True):
        source_embedding = self.embedding_matrix(source_ids) + self.positional_encoding[:source_ids.shape[1]] # pyright: ignore[reportIndexIssue]
        target_embedding = self.embedding_matrix(target_ids) + self.positional_encoding[:target_ids.shape[1]] # pyright: ignore[reportIndexIssue]

        return self.head(self.transformer(source_embedding,
                                          target_embedding,
                                          source_mask=source_mask,
                                          source_key_padding_mask=source_key_padding_mask,
                                          target_mask=target_mask,
                                          target_key_padding_mask=target_key_padding_mask,
                                          memory_mask=memory_mask,
                                          memory_key_padding_mask=memory_key_padding_mask,
                                          target_is_causal=target_is_causal))

    def generate(self,
                 source_ids: torch.Tensor,
                 special_token_ids: dict[str, int],
                 source_mask: Optional[torch.Tensor]=None,
                 source_key_padding_mask: Optional[torch.Tensor]=None,
                 memory_mask: Optional[torch.Tensor]=None,
                 memory_key_padding_mask: Optional[torch.Tensor]=None,
                 beam_width: int=4,
                 length_penalty: float=0.6,
                 extra_length: int=50):
        batch_size, max_input_length = source_ids.shape
        # repeat source_ids and masks to match the size of candidates so we could feed them to transformer
        source_ids = source_ids.repeat_interleave(beam_width, dim=0)

        if source_key_padding_mask is not None:
            source_key_padding_mask = source_key_padding_mask.repeat_interleave(beam_width, dim=0)
        
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(beam_width, dim=0)

        device = next(self.parameters()).device
        max_output_length = max_input_length + extra_length
        candidates = torch.full((batch_size * beam_width, max_output_length), special_token_ids["pad_token"], device=device)
        candidates[:, 0] = special_token_ids["bos_token"]
        score = torch.full((batch_size * beam_width,), float("-inf"), device=device)
        score[::beam_width] = 0

        # auto-regressive steps, the search stops either when the generated length is larger than the input length + extra_length
        # or when all beams are ended with eos
        length = torch.full((batch_size * beam_width, 1), 1, device=device)
        for i in range(1, max_output_length):
            length_mask = (candidates == special_token_ids["eos_token"]).int()
            has_eos = length_mask.any(dim=-1)

            # only stop when all beams are done, finished beams are still fed to the transformer, not efficient
            if torch.all(has_eos):
                break
            
            # predict next token
            last_token_logits: torch.Tensor = self.transformer(self.embedding_matrix(source_ids) + self.positional_encoding[:source_ids.shape[1]], # pyright: ignore[reportIndexIssue]
                                                               self.embedding_matrix(candidates) + self.positional_encoding[:candidates.shape[1]], # pyright: ignore[reportIndexIssue]
                                                               source_mask=source_mask,
                                                               source_key_padding_mask=source_key_padding_mask,
                                                               memory_mask=memory_mask,
                                                               memory_key_padding_mask=memory_key_padding_mask,
                                                               target_is_causal=True)[:, -1, :]
            last_token_prob = torch.log_softmax(last_token_logits, dim=-1)
            
            # calculate score for the next (batch * beam_width * beam_width) possibilities
            top_probs, top_prob_indexes = torch.topk(last_token_prob, k=beam_width, dim=-1)
            # need to add one to the length because argmax returns the position not length and length_mask.shape[-1] does not account for the next generated token
            length = torch.where(has_eos, length_mask.argmax(dim=-1), length_mask.shape[-1]).unsqueeze(dim=1) + 1

            # has_eos doesn't include the new generated token, meaning the first eos of each beam is included in the score
            top_k_next_score = (score.unsqueeze(dim=-1) + torch.where(has_eos.unsqueeze(dim=-1), 0, top_probs))
            normalized_score = top_k_next_score / length ** length_penalty
            flatten_normalized_score = normalized_score.reshape(normalized_score.shape[0] // beam_width, -1)
            _, beam_indexes = torch.topk(flatten_normalized_score, k=beam_width)
            flatten_next_score = top_k_next_score.reshape(top_k_next_score.shape[0] // beam_width, -1)
            score.copy_(torch.gather(flatten_next_score, dim=1, index=beam_indexes).flatten())

            # update candidates with the score
            beam_offset = (torch.arange(0, batch_size) * beam_width).repeat_interleave(beam_width).to(device)
            prefix_beam_index = (beam_offset + beam_indexes.flatten() // beam_width)
            next_token = torch.gather(top_prob_indexes.view((batch_size, -1)), dim=1, index=beam_indexes).view((-1, 1))
            prefix_has_eos = (candidates[prefix_beam_index] == special_token_ids["eos_token"]).any(dim=1, keepdim=True)
            masked_next_token = next_token.masked_fill_(prefix_has_eos, special_token_ids["eos_token"])
            candidates = candidates[prefix_beam_index]
            candidates[:, i] = masked_next_token[:, 0]
        
        # need to re-calculate the length because the candidates are re-ordered after computed length in the loop
        length_mask = (candidates == special_token_ids["eos_token"]).int()
        length = torch.where(length_mask.any(dim=-1), length_mask.argmax(dim=-1) + 1, length_mask.shape[-1])
        score /= length ** length_penalty
        result_score_index = score.view((-1, beam_width)).argmax(dim=1)

        return candidates.view((batch_size, beam_width, -1))[torch.arange(batch_size), result_score_index]


def main():
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    tokenizer.add_special_tokens({
        "bos_token": "<bos>",
    })

    special_token_ids = tokenizer.special_tokens_map
    for key, token in special_token_ids.items():
        special_token_ids[key] = tokenizer.convert_tokens_to_ids(token)

    vocab_size = tokenizer.vocab_size + 1
    hparams = {
        "token_max_length": 512,
        "batch_size": 128,
        "test_ratio": 0.1,
        "d_model": 256,
        "warmup_steps": 400,
        "label_smoothing": 0.1,
        "beta1": 0.9,
        "beta2":0.98,
        "epsilon": 1e-9,
        "epoch": 3,
        "beam_width": 4
    }

    def tokenize(examples):
        return tokenizer(
            [tokenizer.bos_token + e["en"] + tokenizer.eos_token for e in examples["translation"]],
            text_target=[tokenizer.bos_token + e["fr"] + tokenizer.eos_token for e in examples["translation"]],
            truncation=True,
            max_length=hparams["token_max_length"])

    dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr", num_proc=os.cpu_count())["train"].train_test_split(test_size=hparams["test_ratio"], seed=42)
    print(f"Loaded dataset with vocabulary size {vocab_size}")
    tokenize_dataset = dataset.map(tokenize, batched=True)
    tokenize_dataset["train"].set_format(columns=["input_ids", "labels", "attention_mask"])

    # only labels use special ID to indicate paddings
    # because input_ids mask need to stay within [0, vocal_size - 1] to do inner product with embedding matrix
    LABEL_PAD_ID = -100
    train_loader = DataLoader(tokenize_dataset["train"],          # type: ignore
                            batch_size=hparams["batch_size"],
                            num_workers=os.cpu_count() or 1,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=DataCollatorForSeq2Seq(tokenizer, padding="longest", max_length=hparams["token_max_length"], label_pad_token_id=LABEL_PAD_ID))

    tokenize_dataset["test"].set_format(columns=["input_ids", "labels", "attention_mask"])
    test_loader = DataLoader(tokenize_dataset["test"],          # type: ignore
                            batch_size=hparams["batch_size"],
                            num_workers=os.cpu_count() or 1,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=DataCollatorForSeq2Seq(tokenizer, padding="longest", max_length=hparams["token_max_length"], label_pad_token_id=LABEL_PAD_ID))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel(vocab_size, hparams["d_model"], encoder_num_layers=1, decoder_num_layers=1).to(device)

    # since the formula for learning rate in the paper is d^-0.5_model * min(step_num^-0.5, step_num * warmup_steps^-1.5)
    # target/maximum learning rate happens when step_num == warmup_steps
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(hparams["beta1"], hparams["beta2"]), eps=hparams["epsilon"])
    def lr_lambda(step_num: int):
        # step_num starts from 0 in Pytorch but the paper's formula starts from 1
        step_num += 1
        return hparams["d_model"] ** -0.5 * min((step_num + 1) ** -0.5, (step_num + 1) * hparams["warmup_steps"] ** -1.5)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=LABEL_PAD_ID, label_smoothing=hparams["label_smoothing"])

    log_dir = "runs/one-tenth-warmup"
    writer = SummaryWriter(log_dir=log_dir)

    def move_dict_to_device(d: dict[str, torch.Tensor], device: str):
        for key, value in d.items():
            d[key] = value.to(device, non_blocking=True)

    print(f"Train data size: {len(dataset['train'])}")
    print(f"Test data size: {len(dataset['test'])}")
    print(f"Start training on device {device}")

    try:
        for e in range(hparams["epoch"]):
            print(f"---------- epoch {e} ----------")
            model.train()
            for i, batch in enumerate(train_loader):
                move_dict_to_device(batch, device)
                target_ids = batch["labels"][:, :-1]
                label_ids = batch["labels"][:, 1:]
                target_key_padding_mask = target_ids == LABEL_PAD_ID

                with torch.autocast(device, dtype=torch.bfloat16):
                    logits: torch.Tensor = model(batch["input_ids"],
                                                torch.masked_fill(target_ids, target_key_padding_mask, tokenizer.pad_token_id),
                                                source_key_padding_mask=batch["attention_mask"] == 0,
                                                target_key_padding_mask=target_key_padding_mask,
                                                memory_key_padding_mask=batch["attention_mask"] == 0,
                                                target_is_causal=True)
                    
                    # the labels_ids need to use -100 to mark paddings so the loss function knows not to take them into account
                    loss = loss_fn(logits.view((-1, logits.shape[-1])), label_ids.reshape(-1))

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                writer.add_scalar("train/loss", loss.item(), e * len(train_loader) + i)
                print(f"Loss {loss.item():>7f}, [{i + 1}/{len(train_loader)}]")

            model.eval()
            with torch.inference_mode():
                for i, batch in enumerate(test_loader):
                    move_dict_to_device(batch, device)
                    target_ids = batch["labels"][:, :-1]
                    label_ids = batch["labels"][:, 1:]
                    target_key_padding_mask = target_ids == LABEL_PAD_ID

                    with torch.autocast(device, dtype=torch.bfloat16):
                        logits = model(batch["input_ids"],
                                torch.masked_fill(target_ids, target_key_padding_mask, tokenizer.pad_token_id),
                                source_key_padding_mask=batch["attention_mask"] == 0,
                                target_key_padding_mask=target_key_padding_mask,
                                memory_key_padding_mask=batch["attention_mask"] == 0,
                                target_is_causal=True)
                        loss = loss_fn(logits.view((-1, logits.shape[-1])), label_ids.reshape(-1))

                    writer.add_scalar("test/loss", loss.item(), e * len(test_loader) + i)

        print("Finish training")

        MODEL_PATH = Path("models")
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        MODEL_SAVE_PATH = MODEL_PATH / "base_model.pth"
        print(f"Saving model to: {MODEL_SAVE_PATH}")
        torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

        bleu_score = BLEUScore(n_gram=4).to(device)
        model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(test_loader):
                print()
                move_dict_to_device(batch, device)
                key_padding_mask = batch["attention_mask"] == 0
                predict_tokens = model.generate(batch["input_ids"], special_token_ids, source_key_padding_mask=key_padding_mask, memory_key_padding_mask=key_padding_mask)
                padded_labels = torch.masked_fill(batch["labels"][:, 1:], batch["labels"][:, 1:] == LABEL_PAD_ID, tokenizer.pad_token_id)
                predict_sentence = tokenizer.batch_decode(predict_tokens, skip_special_tokens=True)
                target_sentence = tokenizer.batch_decode(padded_labels, skip_special_tokens=True)
                bleu_score.update(predict_sentence, target_sentence)
                print(f"Translation {predict_sentence[0]}, Target: {target_sentence[0]}")

        writer.add_hparams(hparams, {"bleu_score": bleu_score.compute()})
    except Exception as e:
        print("Training failed, cleaning logs...")

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        print(e)
    finally:
        writer.close()

if __name__ == "__main__":
    main()