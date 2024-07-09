"""
Description: In this file we define the dataset class, class for using items in batch, and item class.
    Need a lot refactoring.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pydantic, torch, transformers, pandas
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset
from llm_inference.config.config import Config
from llm_inference.dataset.msg_formatter import MsgFormatterFabric
from llm_inference.type.model_type import ModelType
from llm_inference.type.msg_role_type import MsgRoleType
from pydantic import BaseModel, ConfigDict, Field, computed_field
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


class MsgDatasetItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sentence: str = Field(default="")
    group_id: str | int = Field(default=0)
    device: str = Field(default="cuda")
    tokens: Optional[BatchEncoding] = Field(default=None)

    @computed_field
    def cnt_tokens(self) -> int:
        return self.tokens["input_ids"].shape[1] if self.tokens is not None else 0  # type: ignore


# WARNING: This class is not fully supported yet, not tested and should not be used.
class HfMsgDataset:
    def __init__(
        self,
        messages_df: pd.DataFrame,
        configs: Config,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # We need to set `use_fast` to false if we want use multiprocessing.
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.get_tokenizer(configs.llm_model.llm_url)
        )

        self.device = configs.general.device_type
        self.configs = configs.llm_model.dataset
        self.llm_model_type: ModelType = configs.llm_model.llm_model_type
        self.fmt = MsgFormatterFabric.get_formatter(self.llm_model_type)()
        self.dataset: list[MsgDatasetItem] = self.format_dataset(messages_df)

    @staticmethod
    def get_tokenizer(url: str):
        tokenizer = AutoTokenizer.from_pretrained(
            url, padding_side="left", use_fast=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @staticmethod
    def from_csv(csv_path: Path, configs: Config) -> "HfMsgDataset":
        messages_df = pd.read_csv(csv_path)
        return HfMsgDataset(messages_df, configs)

    @staticmethod
    def from_df_one_sys(
        df: pd.DataFrame, configs: Config, sys_prompt: str
    ) -> "HfMsgDataset":
        sys_df = pd.DataFrame(
            {"role": [MsgRoleType.SYSTEM], "content": [sys_prompt], "group_id": [0]}
        )

        df["role"] = df["role"].apply(lambda x: MsgRoleType(x))  # type: ignore
        formatted_df = pd.DataFrame(columns=df.columns)
        for id, group in df.groupby("group_id"):
            sys_df["group_id"] = id
            group = pd.concat([sys_df, group])
            formatted_df = pd.concat([formatted_df, group])

        return HfMsgDataset(formatted_df, configs)

    def format_dataset(self, messages_df: pd.DataFrame) -> list[MsgDatasetItem]:
        # Apply transformations
        messages_df["role"] = messages_df["role"].apply(
            lambda x: MsgRoleType(x)  # type: ignore
        )

        if "group_id" not in messages_df.columns:
            messages_df["group_id"] = 0

        return [
            MsgDatasetItem(sentence=self.fmt.format_dialog(group), group_id=id)
            for id, group in messages_df.groupby("group_id")
        ]

    def tokenize(self, item: MsgDatasetItem) -> None:
        item.tokens = self.tokenizer(item.sentence, return_tensors="pt")

    def batch_decode(
        self,
        model_output_tokens: list[list[int]] | torch.Tensor,
        orig_cnt_tokens: Optional[int] = None,
    ) -> list[str]:
        if self.tokenizer is None:
            raise ValueError("MsgDataset::batch_decode| Tokenizer is None")

        batch = (
            torch.tensor(model_output_tokens)
            if isinstance(model_output_tokens, list)
            else model_output_tokens
        )
        batch = batch[:, orig_cnt_tokens:]
        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    def decode(self, model_output_tokens: list[int]) -> str:
        if self.tokenizer is None:
            raise ValueError("MsgDataset::decode| Tokenizer is None")

        return self.tokenizer.decode(model_output_tokens)

    def batch_encode(self, batch):
        return self.tokenizer(
            batch["input_sentence"], padding=True, return_tensors="pt"
        ).to("cuda")

    def get_hf_dataset(self) -> Dataset:
        sentence = [item.sentence for item in self.dataset]
        groups_id = [item.group_id for item in self.dataset]
        dataset = Dataset.from_dict({"input_sentence": sentence, "group_id": groups_id})

        return dataset.with_transform(self.batch_encode)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | slice) -> MsgDatasetItem | list[MsgDatasetItem]:
        return self.dataset[index]
