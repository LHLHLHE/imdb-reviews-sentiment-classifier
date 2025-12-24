import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_jsonl(
    path: Path, text_field: str, label_field: str | None
) -> tuple[list[str], np.ndarray | None]:
    path = Path(path)
    texts: list[str] = []
    labels: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(str(obj[text_field]))
            if label_field is not None:
                labels.append(int(obj[label_field]))

    if label_field is None:
        return texts, None
    return texts, np.asarray(labels, dtype=np.int64)


class JsonlBertDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray | None,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def init_dataloader(
    dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 2
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )


@dataclass(frozen=True)
class SplitConfig:
    val_size: float
    random_state: int
    shuffle: bool


class ImdbReviewsDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        train_path: Path | None = None,
        test_path: Path | None = None,
        predict_path: Path | None = None,
        text_field: str = "review",
        label_field: str = "label",
        split: SplitConfig,
        pretrained_model_name: str,
        max_length: int,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.text_field = text_field
        self.label_field = label_field

        self.split = split

        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.tokenizer = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def prepare_data(self) -> None:
        # прогрев кеша
        AutoTokenizer.from_pretrained(self.pretrained_model_name)

    def setup(self, stage: str | None = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        if stage == "fit":
            if self.train_path is None:
                raise ValueError("train_path is required for stage='fit'")

            texts, labels = read_jsonl(self.train_path, self.text_field, self.label_field)
            x_tr, x_val, y_tr, y_val = train_test_split(
                texts,
                labels,
                test_size=self.split.val_size,
                random_state=self.split.random_state,
                shuffle=self.split.shuffle,
                stratify=labels,
            )
            self.train_dataset = JsonlBertDataset(x_tr, y_tr, self.tokenizer, self.max_length)
            self.val_dataset = JsonlBertDataset(x_val, y_val, self.tokenizer, self.max_length)
        elif stage == "validate":
            if self.train_path is None:
                raise ValueError("train_path is required for stage='validate'")

            texts, labels = read_jsonl(self.train_path, self.text_field, self.label_field)
            _, x_val, _, y_val = train_test_split(
                texts,
                labels,
                test_size=self.split.val_size,
                random_state=self.split.random_state,
                shuffle=self.split.shuffle,
                stratify=labels,
            )
            self.val_dataset = JsonlBertDataset(x_val, y_val, self.tokenizer, self.max_length)
        elif stage == "test":
            if self.test_path is None:
                raise ValueError("test_path is required for stage='test'")

            texts, labels = read_jsonl(self.test_path, self.text_field, self.label_field)
            self.test_dataset = JsonlBertDataset(texts, labels, self.tokenizer, self.max_length)
        elif stage == "predict":
            if self.predict_path is None:
                raise ValueError("predict_path is required for stage='predict'")

            texts, _ = read_jsonl(self.predict_path, self.text_field, label_field=None)
            self.predict_dataset = JsonlBertDataset(
                texts, labels=None, tokenizer=self.tokenizer, max_length=self.max_length
            )

    def train_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.train_dataset, self.train_batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.val_dataset, self.eval_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.test_dataset, self.eval_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> DataLoader:
        return init_dataloader(
            self.predict_dataset, self.eval_batch_size, shuffle=False, num_workers=self.num_workers
        )
