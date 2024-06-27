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
from typing import Optional
from pathlib import Path

import pandas as pd
import torch
from llm_inference.config.config import Config
from llm_inference.dataset.msg_formatter import MsgFormatterFabric
from llm_inference.type.msg_role_type import MsgRoleType
from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch.utils.data import Dataset
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


# TODO: Refactor this class
class MsgDatasetBatch:
    def __init__(
        self,
        batch: list[MsgDatasetItem],
        tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast],
    ) -> None:
        self.logger = logging.getLogger(__name__)
        # self.logger.debug("MsgDatasetBatch::init| Object created.")

        self.device = batch[0].device
        self.tokenizer = tokenizer if tokenizer is not None else None
        # TODO: Find how we can solve this.
        # We have problem with this class and using dataloader.
        # We can use tokenizer only before dataloader, because it using multiprocessing.
        # https://github.com/huggingface/transformers/issues/5486
        # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning/72926996#72926996
        self.batch_data: Optional[BatchEncoding] = (
            self._get_batch_tokens(batch).to(self.device)
            if self.tokenizer is not None
            else None
        )

        # NOTE: When we use data in batch, we need to apply padding.
        # Padding is added extra tokens at the end and at the beginning.
        # NOTE: With padding we need only one count of tokens, because it the same size for all items in batch.
        self.cnt_tokens_batch = (
            self.batch_data.data["input_ids"].shape[1]
            if self.batch_data is not None
            else 0
        )

        self.sentences = [item.sentence for item in batch]
        self.len_seq_batch: list[int] = [len(sentence) for sentence in self.sentences]
        self.groups_id = [item.group_id for item in batch]

    def _get_batch_tokens(self, batch: list[MsgDatasetItem]):
        if self.tokenizer is None:
            raise ValueError("MsgDatasetBatch::_get_batch_tokens| Tokenizer is None")

        # TODO: Check that we can use tokenizer.pad here
        # NOTE: It can be okay, but we tokenize twice. Need to refactor this.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        sequence: list[str] = [item.sentence for item in batch]
        return self.tokenizer(sequence, return_tensors="pt", padding=True)

    def to(self, device: str):
        if self.batch_data is None:
            raise ValueError("MsgDatasetBatch::to| Batch data is None")

        self.device = device
        self.batch_data = self.batch_data.to(device)

    # TODO: Add support for pin_memory
    # def pin_memory(self):
    #     self.batch_data = self.batch_data.data["input_ids"].pin_memory()
    #     return self


# WARNING: This class is not fully supported yet, not tested and should not be used.
class MsgDataset(Dataset):
    def __init__(
        self,
        messages_df: pd.DataFrame,
        configs: Config,
        skip_tokenizer: bool = False,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        # We need to set `use_fast` to false if we want use multiprocessing.
        self.tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = (
            self.get_tokenizer(configs.llm_model.llm_url)
            if not skip_tokenizer
            else None
        )

        self.device = configs.general.device_type
        self.configs = configs.llm_model.dataset
        self.llm_model_type = configs.llm_model.llm_model_type
        self.fmt = MsgFormatterFabric.get_formatter(self.llm_model_type)()
        self.dataset = self.format_dataset(messages_df)

        if self.tokenizer is not None:
            self.tokenize(self.tokenizer)

    @staticmethod
    def get_tokenizer(url: str):
        return AutoTokenizer.from_pretrained(url, padding_side="left", use_fast=True)

    @staticmethod
    def from_csv(
        csv_path: Path, configs: Config, skip_tokenizer: bool = False
    ) -> "MsgDataset":
        messages_df = pd.read_csv(csv_path)
        return MsgDataset(messages_df, configs, skip_tokenizer=skip_tokenizer)

    @staticmethod
    def from_df_one_sys(
        df: pd.DataFrame, configs: Config, sys_prompt: str
    ) -> "MsgDataset":
        sys_df = pd.DataFrame(
            {"role": [MsgRoleType.SYSTEM], "content": [sys_prompt], "group_id": [0]}
        )

        df["role"] = df["role"].apply(lambda x: MsgRoleType(x))  # type: ignore
        formatted_df = pd.DataFrame(columns=df.columns)
        for id, group in df.groupby("group_id"):
            sys_df["group_id"] = id
            group = pd.concat([sys_df, group])
            formatted_df = pd.concat([formatted_df, group])

        return MsgDataset(formatted_df, configs)

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

    # NOTE: Maybe made this method static?
    def tokenize(self, tokenizer) -> None:
        for item in self.dataset:
            item.tokens = tokenizer(item.sentence, return_tensors="pt")

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
        tmp = self.tokenizer.batch_decode(model_output_tokens)
        for item in tmp:
            print(item)

        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    def decode(self, model_output_tokens: list[int]) -> str:
        if self.tokenizer is None:
            raise ValueError("MsgDataset::decode| Tokenizer is None")

        return self.tokenizer.decode(model_output_tokens)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> MsgDatasetItem:
        return self.dataset[index]
