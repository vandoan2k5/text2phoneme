from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import load_runtime_config
from dataset import (
    ByteLevelBPETokenizer,
    EspeakTokenizer,
    InterleavedPhonemizer,
    PhonemeVocabulary,
    strip_language_tags,
)
from model.net import Text2PhonemeTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/best.pt")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"] if args.config is None else load_runtime_config(args.config)

    text_tokenizer = ByteLevelBPETokenizer.load(str(Path(config["artifacts"]["dir"]) / "text_tokenizer.json"))
    phoneme_vocab = PhonemeVocabulary.from_lang_token_files(
        {
            config["data"]["vi_lang"]: config["data"]["phoneme_token_files"][0],
            config["data"]["en_lang"]: config["data"]["phoneme_token_files"][1],
        }
    )
    vi_tokenizer = EspeakTokenizer(phoneme_vocab, lang=config["data"]["vi_lang"])
    en_tokenizer = EspeakTokenizer(phoneme_vocab, lang=config["data"]["en_lang"])
    phonemizer = InterleavedPhonemizer(vi_tokenizer=vi_tokenizer, en_tokenizer=en_tokenizer)

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
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    source_ids_list = text_tokenizer.encode(strip_language_tags(args.text))[: config["model"]["max_source_length"]]
    if source_ids_list[-1] != text_tokenizer.eos_id:
        source_ids_list[-1] = text_tokenizer.eos_id
    source_ids = torch.tensor([source_ids_list], dtype=torch.long, device=device)
    source_mask = source_ids.eq(text_tokenizer.pad_id)
    generated = model.greedy_decode(
        source_ids=source_ids,
        source_padding_mask=source_mask,
        bos_id=phonemizer.vocab.bos_id,
        eos_id=phonemizer.vocab.eos_id,
    )
    print(phonemizer.decode_ids(generated[0].tolist()))


if __name__ == "__main__":
    main()
