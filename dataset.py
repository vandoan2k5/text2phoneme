from __future__ import annotations

import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import Dataset as TorchDataset
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, processors, trainers

try:
    from piper_phonemize import phonemize_espeak
except Exception as ex:  # pragma: no cover - import error surfaces at runtime
    raise RuntimeError(
        f"{ex}\nPlease run\n"
        "pip install piper_phonemize -f "
        "https://k2-fsa.github.io/icefall/piper_phonemize.html"
    ) from ex


INTERLEAVED_PATTERN = re.compile(r"(<en>.*?</en>)", flags=re.DOTALL)
EN_OPEN_TAG = "<en>"
EN_CLOSE_TAG = "</en>"


def split_interleaved_text(text: str) -> List[tuple[str, str]]:
    parts = INTERLEAVED_PATTERN.split(text)
    segments: List[tuple[str, str]] = []
    for part in parts:
        if not part:
            continue
        if part.startswith(EN_OPEN_TAG) and part.endswith(EN_CLOSE_TAG):
            content = part[len(EN_OPEN_TAG) : -len(EN_CLOSE_TAG)]
            if content:
                segments.append(("en", content))
        elif part:
            segments.append(("vi", part))
    return segments


def strip_language_tags(text: str) -> str:
    return text.replace(EN_OPEN_TAG, "").replace(EN_CLOSE_TAG, "")


class Tokenizer(ABC):
    @abstractmethod
    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        raise NotImplementedError

    @abstractmethod
    def tokens_to_token_ids(self, tokens: List[List[str]]) -> List[List[int]]:
        raise NotImplementedError


@dataclass
class PhonemeVocabulary:
    token2id: Dict[str, int]
    id2symbol: Dict[int, str]

    @classmethod
    def from_lang_token_files(cls, lang_token_files: Dict[str, str]) -> "PhonemeVocabulary":
        special_tokens = ["_", "^", "$"]
        token2id: Dict[str, int] = {token: idx for idx, token in enumerate(special_tokens)}
        id2symbol: Dict[int, str] = {idx: token for idx, token in enumerate(special_tokens)}

        next_id = len(special_tokens)
        for lang, token_file in lang_token_files.items():
            with open(token_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    token, _ = line.rstrip().split("\t")
                    if token in special_tokens:
                        continue
                    namespaced_token = f"{lang}:{token}"
                    if namespaced_token in token2id:
                        continue
                    token2id[namespaced_token] = next_id
                    id2symbol[next_id] = token
                    next_id += 1
        return cls(token2id=token2id, id2symbol=id2symbol)

    @property
    def pad_id(self) -> int:
        return self.token2id["_"]

    @property
    def bos_id(self) -> int:
        return self.token2id["^"]

    @property
    def eos_id(self) -> int:
        return self.token2id["$"]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def id2token(self) -> Dict[int, str]:
        return self.id2symbol


class EspeakTokenizer(Tokenizer):
    def __init__(self, vocab: PhonemeVocabulary, lang: str = "en-us"):
        self.lang = lang
        self.vocab = vocab
        self.token2id = vocab.token2id
        self.pad_id = vocab.pad_id
        self.bos_id = vocab.bos_id
        self.eos_id = vocab.eos_id
        self.vocab_size = vocab.vocab_size

    def g2p(self, text: str) -> List[str]:
        try:
            tokens = phonemize_espeak(text, self.lang)
            return reduce(lambda x, y: x + y, tokens, [])
        except Exception as ex:
            logging.warning("Tokenization of %s text failed: %s", self.lang, ex)
            return []

    def texts_to_token_ids(self, texts: List[str]) -> List[List[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(self, texts: List[str]) -> List[List[str]]:
        return [self.g2p(text) for text in texts]

    def tokens_to_token_ids(self, tokens_list: List[List[str]]) -> List[List[int]]:
        token_ids_list = []
        for tokens in tokens_list:
            token_ids = []
            for token in tokens:
                namespaced_token = f"{self.lang}:{token}"
                if namespaced_token not in self.token2id:
                    logging.debug("Skip OOV phoneme token %s for lang %s", token, self.lang)
                    continue
                token_ids.append(self.token2id[namespaced_token])
            token_ids_list.append(token_ids)
        return token_ids_list


class InterleavedPhonemizer:
    def __init__(self, vi_tokenizer: EspeakTokenizer, en_tokenizer: EspeakTokenizer):
        self.vi_tokenizer = vi_tokenizer
        self.en_tokenizer = en_tokenizer
        self.vocab = vi_tokenizer.vocab

    def text_to_phoneme_ids(self, text: str) -> List[int]:
        phoneme_ids: List[int] = [self.vocab.bos_id]
        for lang, segment in split_interleaved_text(text):
            segment_ids = (
                self.en_tokenizer.texts_to_token_ids([segment])[0]
                if lang == "en"
                else self.vi_tokenizer.texts_to_token_ids([segment])[0]
            )
            phoneme_ids.extend(segment_ids)
        phoneme_ids.append(self.vocab.eos_id)
        return phoneme_ids

    def decode_ids(self, token_ids: Sequence[int]) -> str:
        id2token = self.vocab.id2token
        pieces = []
        for idx in token_ids:
            if idx in {self.vocab.pad_id, self.vocab.bos_id, self.vocab.eos_id}:
                continue
            pieces.append(id2token.get(idx, ""))
        return "".join(pieces)


class ByteLevelBPETokenizer:
    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    special_tokens = [pad_token, bos_token, eos_token, unk_token]

    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id(self.pad_token)
        self.bos_id = self.tokenizer.token_to_id(self.bos_token)
        self.eos_id = self.tokenizer.token_to_id(self.eos_token)
        self.unk_id = self.tokenizer.token_to_id(self.unk_token)
        if None in {self.pad_id, self.bos_id, self.eos_id, self.unk_id}:
            raise ValueError("Tokenizer is missing required special tokens.")

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
    ) -> "ByteLevelBPETokenizer":
        tokenizer = HFTokenizer(models.BPE(unk_token=cls.unk_token))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator((strip_language_tags(text) for text in texts), trainer=trainer)
        return cls(tokenizer=tokenizer)

    def encode(self, text: str) -> List[int]:
        encoding = self.tokenizer.encode(strip_language_tags(text))
        return [self.bos_id, *encoding.ids, self.eos_id]

    def decode(self, token_ids: Sequence[int]) -> str:
        filtered_ids = [idx for idx in token_ids if idx not in {self.pad_id, self.bos_id, self.eos_id}]
        return self.tokenizer.decode(filtered_ids, skip_special_tokens=False)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        return cls(tokenizer=HFTokenizer.from_file(path))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def load_named_datasets(
    data_root: str,
    dataset_names: Sequence[str],
    split: str = "train",
    limit_per_dataset: Optional[int] = None,
    dataset_sample_limits: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Dataset:
    loaded = []
    for name in dataset_names:
        dataset = load_dataset(str(Path(data_root) / name), split=split)
        dataset_limit = None
        if dataset_sample_limits is not None and name in dataset_sample_limits:
            dataset_limit = dataset_sample_limits[name]
        elif limit_per_dataset is not None:
            dataset_limit = limit_per_dataset

        if dataset_limit is not None:
            limit = min(dataset_limit, len(dataset))
            dataset = dataset.shuffle(seed=seed).select(range(limit))
        loaded.append(dataset)
    if not loaded:
        raise ValueError("No datasets were loaded.")
    return loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)


def train_eval_split(
    dataset: Dataset,
    eval_ratio: float,
    seed: int,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> tuple[Dataset, Dataset]:
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]
    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
    return train_ds, eval_ds


class Text2PhonemeTorchDataset(TorchDataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        text_tokenizer: CharTextTokenizer,
        phonemizer: InterleavedPhonemizer,
        max_source_length: int,
        max_target_length: int,
    ):
        self.dataset = hf_dataset
        self.text_tokenizer = text_tokenizer
        self.phonemizer = phonemizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        text = self.dataset[index]["text"]
        source_ids = self.text_tokenizer.encode(text)[: self.max_source_length]
        if source_ids[-1] != self.text_tokenizer.eos_id:
            source_ids[-1] = self.text_tokenizer.eos_id

        target_ids = self.phonemizer.text_to_phoneme_ids(text)[: self.max_target_length]
        if target_ids[-1] != self.phonemizer.vocab.eos_id:
            target_ids[-1] = self.phonemizer.vocab.eos_id

        return {
            "text": text,
            "source_ids": source_ids,
            "target_ids": target_ids,
        }


@dataclass
class Batch:
    source_ids: torch.Tensor
    source_padding_mask: torch.Tensor
    target_input_ids: torch.Tensor
    target_output_ids: torch.Tensor
    target_padding_mask: torch.Tensor
    texts: List[str]


class Text2PhonemeCollator:
    def __init__(self, text_pad_id: int, phoneme_pad_id: int):
        self.text_pad_id = text_pad_id
        self.phoneme_pad_id = phoneme_pad_id

    def __call__(self, features: List[Dict[str, List[int]]]) -> Batch:
        texts = [feature["text"] for feature in features]
        src_tensors = [torch.tensor(feature["source_ids"], dtype=torch.long) for feature in features]
        tgt_tensors = [torch.tensor(feature["target_ids"], dtype=torch.long) for feature in features]

        source_ids = torch.nn.utils.rnn.pad_sequence(
            src_tensors,
            batch_first=True,
            padding_value=self.text_pad_id,
        )
        target_ids = torch.nn.utils.rnn.pad_sequence(
            tgt_tensors,
            batch_first=True,
            padding_value=self.phoneme_pad_id,
        )

        target_input_ids = target_ids[:, :-1]
        target_output_ids = target_ids[:, 1:]

        return Batch(
            source_ids=source_ids,
            source_padding_mask=source_ids.eq(self.text_pad_id),
            target_input_ids=target_input_ids,
            target_output_ids=target_output_ids,
            target_padding_mask=target_input_ids.eq(self.phoneme_pad_id),
            texts=texts,
        )


def sample_texts_for_vocab(dataset: Dataset, max_samples: int, seed: int) -> List[str]:
    if len(dataset) <= max_samples:
        return list(dataset["text"])
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    sampled = dataset.select(indices)
    return list(sampled["text"])
