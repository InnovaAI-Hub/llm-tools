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

from pathlib import Path
from typing import Optional, override

import pandas as pd
import torch
import logging
from datasets import Dataset
from llm_tools.config.config import Config
from llm_tools.dataset.msg_dataset import AbstractMsgDataset, MsgDatasetItem
from llm_tools.type.msg_role_type import MsgRoleType
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


# WARNING: This class is not fully supported yet, not tested and should not be used.
class HfMsgDataset(AbstractMsgDataset):
    def __init__(
        self,
        messages_df: pd.DataFrame,
        configs: Config,
    ) -> None:
        super().__init__(messages_df, configs)
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self.get_tokenizer(
                configs.llm_model.llm_url,
                pad_token_id=configs.llm_model.pad_token_id,
                pad_token=configs.llm_model.pad_token,
            )
        )
        self.dataset = self.format_dataset(messages_df=messages_df)

    @staticmethod
    def get_tokenizer(
        url: str,
        set_pad_token: bool = True,
        pad_token: Optional[str] = None,
        pad_token_id: Optional[int] = None,
    ):
        logger = logging.getLogger(__name__)

        tokenizer = AutoTokenizer.from_pretrained(
            url, padding_side="left", use_fast=True
        )

        if set_pad_token and not tokenizer.pad_token:
            logger.debug(
                f"HfMsgDataset::get_tokenizer| Set pad token.\
                    Current pad_token: ({tokenizer.pad_token}, {tokenizer.pad_token_id}),\
                    Eos_token: ({tokenizer.eos_token}, {tokenizer.eos_token_id})."
            )

            tokenizer.pad_token = (
                pad_token if pad_token is not None else tokenizer.eos_token
            )

            tokenizer.pad_token_id = (
                pad_token_id if pad_token_id is not None else tokenizer.eos_token_id
            )

        return tokenizer

    @staticmethod
    def prepare_to_train(messages_df: pd.DataFrame, configs: Config) -> Dataset:
        # First we need to split dialogs. For every group we need to create the
        # subgroups like this: sys + usr, sys + usr + assist + usr

        new_dataset: list[pd.DataFrame] = []
        answers: list[pd.DataFrame] = []

        for group_id, group in messages_df.groupby("group_id"):
            formatted = [group[:i] for i in range(2, group.shape[0], 2)]
            # TODO: Move to enum, some role can have a different name.
            answers.append(
                group.loc[group["role"] == "assistant"][["group_id", "content"]]
            )

            formatted = [df.astype(str) for df in formatted]
            for i, formatted_group in enumerate(formatted):
                formatted_group.loc[:, "group_id"] = f"{group_id}.{i}"

            new_dataset.extend(formatted)

        formatted_df = HfMsgDataset(pd.concat(new_dataset), configs)
        formatted_df.add_valid_data(pd.concat(answers)["content"].to_list())
        return formatted_df.get_hf_dataset()

    @staticmethod
    def from_csv(csv_path: Path, configs: Config) -> "HfMsgDataset":
        messages_df = pd.read_csv(csv_path)
        return HfMsgDataset(messages_df, configs)

    @staticmethod
    def from_df_one_sys(
        df: pd.DataFrame, configs: Config, sys_prompt: str
    ) -> "HfMsgDataset":
        sys_df = pd.DataFrame(
            {
                "role": [MsgRoleType.SYSTEM.value],
                "content": [sys_prompt],
                "group_id": [0],
            }
        )

        # df["role"] = df["role"].apply(lambda x: MsgRoleType(x))  # type: ignore
        formatted_df = pd.DataFrame(columns=df.columns)
        for id, group in df.groupby("group_id"):
            sys_df["group_id"] = id
            group = pd.concat([sys_df, group])
            formatted_df = (
                group if formatted_df.empty else pd.concat([formatted_df, group])
            )

        return HfMsgDataset(formatted_df, configs)

    def _format_group(self, group_id: str | int, group: pd.DataFrame) -> MsgDatasetItem:
        prompt: str = self.tokenizer.apply_chat_template(
            group.to_dict("records"),  # type: ignore
            add_generation_prompt=self.configs.add_generation_prompt,
            tokenize=False,
        )  # type: ignore

        return MsgDatasetItem(sentence=prompt, group_id=group_id)

    @override
    def format_dataset(self, messages_df: pd.DataFrame) -> list[MsgDatasetItem]:
        # Apply transformations.
        # NOTE: Maybe need to remove it?
        # messages_df["role"] = messages_df["role"].apply(
        #     lambda x: MsgRoleType(x)  # type: ignore
        # )

        if "group_id" not in messages_df.columns:
            messages_df["group_id"] = 0

        return [
            self._format_group(group_id=group_id, group=group)  # type: ignore
            for group_id, group in messages_df.groupby("group_id")
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
        # Check that we can remove padding and `to`. And use datacollator for it.
        model_input = self.tokenizer(batch["input_sentence"], padding=True)
        labels = self.tokenizer(batch["valid"], padding=True)

        skip_token = -100
        pad_token = self.tokenizer.pad_token_id
        labels["input_ids"] = [
            [label if label != pad_token else skip_token for label in label_seq]
            for label_seq in labels["input_ids"]
        ]

        model_input["labels"] = labels["input_ids"]
        return model_input

    def add_valid_data(self, valid_data: list[str]):
        assert len(self.dataset) == len(valid_data)
        # TODO: Add merge by group_id
        for i, item in enumerate(valid_data):
            self.dataset[i].valid = item

    # TODO: Refactor this. It's wrong usage.
    def get_hf_dataset(self) -> Dataset:
        sentence = [item.sentence for item in self.dataset]
        groups_id = [item.group_id for item in self.dataset]
        valid = [item.valid for item in self.dataset]
        dataset = Dataset.from_dict(
            {
                "input_sentence": sentence,
                "group_id": groups_id,
                "valid": valid,
            }
        )

        return dataset.map(
            self.batch_encode,
            batched=True,
            remove_columns=["input_sentence", "valid", "group_id"],
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | slice) -> MsgDatasetItem | list[MsgDatasetItem]:
        return self.dataset[index]
