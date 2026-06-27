from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import load_runtime_config
from dataset import (
    ByteLevelBPETokenizer,
    EspeakTokenizer,
    InterleavedPhonemizer,
    PhonemeVocabulary,
    Text2PhonemeCollator,
    Text2PhonemeTorchDataset,
    load_named_datasets,
    sample_texts_for_vocab,
    train_eval_split,
)
from model.net import Text2PhonemeTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train.yaml")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_components(config: Dict[str, Any]):
    seed = config["training"]["seed"]
    dataset = load_named_datasets(
        data_root=config["data"]["root"],
        dataset_names=config["data"]["dataset_names"],
        split="train",
        limit_per_dataset=config["data"].get("limit_per_dataset"),
        dataset_sample_limits=config["data"].get("dataset_sample_limits"),
        seed=seed,
    )
    train_ds, eval_ds = train_eval_split(
        dataset=dataset,
        eval_ratio=config["data"]["eval_ratio"],
        seed=seed,
        max_train_samples=config["data"].get("max_train_samples"),
        max_eval_samples=config["data"].get("max_eval_samples"),
    )

    vocab_texts = sample_texts_for_vocab(
        train_ds,
        max_samples=config["data"]["vocab_sample_size"],
        seed=seed,
    )
    text_tokenizer = ByteLevelBPETokenizer.from_texts(
        vocab_texts,
        vocab_size=config["data"].get("text_tokenizer_vocab_size", 4096),
        min_frequency=config["data"].get("text_tokenizer_min_frequency", 2),
    )

    artifact_dir = Path(config["artifacts"]["dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    text_tokenizer.save(str(artifact_dir / "text_tokenizer.json"))

    phoneme_vocab = PhonemeVocabulary.from_lang_token_files(
        {
            config["data"]["vi_lang"]: config["data"]["phoneme_token_files"][0],
            config["data"]["en_lang"]: config["data"]["phoneme_token_files"][1],
        }
    )
    vi_tokenizer = EspeakTokenizer(phoneme_vocab, lang=config["data"]["vi_lang"])
    en_tokenizer = EspeakTokenizer(phoneme_vocab, lang=config["data"]["en_lang"])
    phonemizer = InterleavedPhonemizer(vi_tokenizer=vi_tokenizer, en_tokenizer=en_tokenizer)

    train_torch_ds = Text2PhonemeTorchDataset(
        hf_dataset=train_ds,
        text_tokenizer=text_tokenizer,
        phonemizer=phonemizer,
        max_source_length=config["model"]["max_source_length"],
        max_target_length=config["model"]["max_target_length"],
    )
    eval_torch_ds = Text2PhonemeTorchDataset(
        hf_dataset=eval_ds,
        text_tokenizer=text_tokenizer,
        phonemizer=phonemizer,
        max_source_length=config["model"]["max_source_length"],
        max_target_length=config["model"]["max_target_length"],
    )
    collator = Text2PhonemeCollator(
        text_pad_id=text_tokenizer.pad_id,
        phoneme_pad_id=phoneme_vocab.pad_id,
    )

    return train_torch_ds, eval_torch_ds, text_tokenizer, phonemizer, collator


def build_model(config: Dict[str, Any], text_tokenizer: ByteLevelBPETokenizer, phonemizer: InterleavedPhonemizer):
    model = Text2PhonemeTransformer(
        src_vocab_size=text_tokenizer.vocab_size,
        tgt_vocab_size=phonemizer.vocab.vocab_size,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_encoder_layers=config["model"]["num_encoder_layers"],
        num_decoder_layers=config["model"]["num_decoder_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        attention_dropout=config["model"].get("attention_dropout", config["model"]["dropout"]),
        residual_dropout=config["model"].get("residual_dropout", config["model"]["dropout"]),
        max_source_length=config["model"]["max_source_length"],
        max_target_length=config["model"]["max_target_length"],
        src_pad_id=text_tokenizer.pad_id,
        tgt_pad_id=phonemizer.vocab.pad_id,
    )
    return model


def move_batch_to_device(batch, device: torch.device):
    batch.source_ids = batch.source_ids.to(device, non_blocking=True)
    batch.source_padding_mask = batch.source_padding_mask.to(device, non_blocking=True)
    batch.target_input_ids = batch.target_input_ids.to(device, non_blocking=True)
    batch.target_output_ids = batch.target_output_ids.to(device, non_blocking=True)
    batch.target_padding_mask = batch.target_padding_mask.to(device, non_blocking=True)
    return batch


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        output = model(
            source_ids=batch.source_ids,
            target_input_ids=batch.target_input_ids,
            source_padding_mask=batch.source_padding_mask,
            target_padding_mask=batch.target_padding_mask,
            labels=batch.target_output_ids,
        )
        total_loss += output["loss"].mean().item()
        total_steps += 1
    return total_loss / max(total_steps, 1)


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config)
    set_seed(config["training"]["seed"])

    train_ds, eval_ds, text_tokenizer, phonemizer, collator = prepare_components(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, text_tokenizer, phonemizer)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_eval_loss = math.inf
    artifact_dir = Path(config["artifacts"]["dir"])
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        total_loss = 0.0

        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available(), dtype=torch.float16):
                output = model(
                    source_ids=batch.source_ids,
                    target_input_ids=batch.target_input_ids,
                    source_padding_mask=batch.source_padding_mask,
                    target_padding_mask=batch.target_padding_mask,
                    labels=batch.target_output_ids,
                )
                loss = output["loss"].mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

        train_loss = total_loss / max(len(train_loader), 1)
        eval_loss = evaluate(model, eval_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss})

        checkpoint = {
            "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "config": config,
            "history": history,
        }
        torch.save(checkpoint, artifact_dir / "last.pt")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(checkpoint, artifact_dir / "best.pt")

        print(
            json.dumps(
                {"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss},
                ensure_ascii=False,
            )
        )

    (artifact_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
