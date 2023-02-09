from typing import Union, List
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

from transformers import PreTrainedTokenizer
from transformers import InputFeatures as TransformersInputFeatures
import pytorch_lightning as pl


class Encoder:
    def __init__(self, tokenizer, max_seq_len=300):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def encode(self, df: pd.DataFrame):
        feature_list = list(df.apply(self.encode_row, axis=1))
        return feature_list

    def encode_row(self, row: pd.Series):
        inputs = self.encode_text(row["full_text"])
        labels = torch.tensor(row["labels"], dtype=torch.long)
        sample = {
            "input_ids": torch.squeeze(inputs["input_ids"]),
            "attention_mask": torch.squeeze(inputs["attention_mask"]),
            "label": labels,
        }

        return TransformersInputFeatures(**sample)

    def encode_text(self, text: Union[str, List[str]]):
        is_batched = isinstance(text, (list, tuple))

        if is_batched:
            text = [t for t in text]
        else:
            pass

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        return inputs


class MultiClassDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizer, features: List[TransformersInputFeatures]
    ):
        """A class that holds the entire dataset in memory.
        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer
            features (List[InputFeatures]): InputFeature is a wrapper for a single example
            https://huggingface.co/transformers/main_classes/processors.html#transformers.data.processors.utils.InputFeatures
        """
        self.tokenizer = tokenizer
        self.input_ids = [f.input_ids for f in features]
        self.attention_mask = [f.attention_mask for f in features]

        self.labels = [f.label for f in features]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
        }


class MultiClassDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        tokenizer: PreTrainedTokenizer,
        encoder,
        batch_size=8,
        **kwargs
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.batch_size = batch_size

        self.train_df = train_df
        self.val_df = val_df

    def setup(self, stage=None):
        train_features = self.encoder.encode(self.train_df)
        val_features = self.encoder.encode(self.val_df)

        self.train_dataset = MultiClassDataset(self.tokenizer, train_features)
        self.val_dataset = MultiClassDataset(self.tokenizer, val_features)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, pin_memory=True)
