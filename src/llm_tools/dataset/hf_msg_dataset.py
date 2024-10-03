"""
Description: In this file we define the dataset class, class for using items in batch, and item class.
    Need a lot refactoring.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 03.10.2024
Version: 0.3
Python Version: 3.12
Dependencies: pydantic, torch, transformers, pandas
"""

from pathlib import Path
from typing import Optional, override
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import deprecated

import pandas as pd
import torch
import logging
from datasets import Dataset
from llm_tools.config.config import Config
from llm_tools.config.model_config import ModelConfigLLM
from llm_tools.dataset.msg_dataset import AbstractMsgDataset, MsgDatasetItem
from llm_tools.type.msg_role_type import MsgRoleType
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


# WARNING: This class is not fully supported yet, not tested and should not be used.
class HfMsgDataset(AbstractMsgDataset):
    def __init__(
        self,
        messages_df: pd.DataFrame,
        configs: Optional[Config] = None,
        tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None,
    ) -> None:
        """
        Constructor for HfMsgDataset class.

        Args:
            messages_df (pd.DataFrame): The DataFrame with messages data.
            configs (Optional[Config]): The Config object with settings for the model.
            tokenizer (Optional[PreTrainedTokenizer|PreTrainedTokenizerFast]): The tokenizer for the model.

        Returns:
            None
        """
        super().__init__(messages_df, configs)
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self._check_tokenizer_initialization(
                configs.llm_model if configs is not None else None,
                tokenizer,
            )
        )
        self.dataset = self.format_dataset(messages_df=messages_df)

    def _check_tokenizer_initialization(
        self,
        llm_model_conf: Optional[ModelConfigLLM],
        tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast],
    ):
        """
        This method checks if the tokenizer is already initialized and if so, returns it.
        If the tokenizer is not initialized, it will try to initialize it with the given
        `llm_model_conf` and `tokenizer` parameters.

        Args:
            llm_model_conf (Optional[ModelConfigLLM]): The ModelConfigLLM object with settings for the model.
            tokenizer (Optional[PreTrainedTokenizer|PreTrainedTokenizerFast]): The tokenizer for the model.

        Raises:
            ValueError: If both `llm_model_conf` and `tokenizer` are None.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The initialized tokenizer.
        """
        if tokenizer is not None:
            return tokenizer

        if llm_model_conf is None:
            raise ValueError(
                "HfMsgDataset::init_tokenizer| `llm_model_conf` is None and `tokenizer` is None"
            )

        tokenizer_path = (
            llm_model_conf.llm_url
            if llm_model_conf.tokenizer_url is None
            else llm_model_conf.tokenizer_url
        )

        return self.get_tokenizer(
            tokenizer_path,
            pad_token_id=llm_model_conf.pad_token_id,
            pad_token=llm_model_conf.pad_token,
            max_length=llm_model_conf.max_tokens,
        )

    @staticmethod
    def setup_tokenizer(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        set_pad_token: bool = True,
        pad_token: Optional[str] = None,
        pad_token_id: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        Setup the tokenizer with the given parameters.

        Args:
            tokenizer (PreTrainedTokenizer|PreTrainedTokenizerFast): The tokenizer to setup.
            set_pad_token (bool, optional): Whether to set the pad token. Defaults to True.
            pad_token (str, optional): The pad token. Defaults to None.
            pad_token_id (int, optional): The pad token id. Defaults to None.
            max_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            PreTrainedTokenizer|PreTrainedTokenizerFast: The setup tokenizer.
        """
        logger = logging.getLogger(__name__)

        if max_length is not None:
            tokenizer.model_max_length = max_length

        if set_pad_token and not tokenizer.pad_token:
            logger.debug(
                "HfMsgDataset::get_tokenizer| Set pad token."
                f"Current pad_token: ({tokenizer.pad_token}, {tokenizer.pad_token_id}),"
                f"Eos_token: ({tokenizer.eos_token}, {tokenizer.eos_token_id})."
            )

            tokenizer.pad_token = (
                pad_token if pad_token is not None else tokenizer.eos_token
            )

            tokenizer.pad_token_id = (
                pad_token_id if pad_token_id is not None else tokenizer.eos_token_id
            )

        return tokenizer

    @staticmethod
    def get_tokenizer(
        url: str,
        set_pad_token: bool = True,
        pad_token: Optional[str] = None,
        pad_token_id: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        """
        Get the tokenizer from the given `url`.

        Args:
            url (str): The URL to the tokenizer.
            set_pad_token (bool, optional): Whether to set the pad token. Defaults to True.
            pad_token (str, optional): The pad token. Defaults to None.
            pad_token_id (int, optional): The pad token id. Defaults to None.
            max_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            PreTrainedTokenizer|PreTrainedTokenizerFast: The setup tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            url,
            padding_side="left",
            use_fast=True,
        )

        return HfMsgDataset.setup_tokenizer(
            tokenizer=tokenizer,
            set_pad_token=set_pad_token,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            max_length=max_length,
        )

    @staticmethod
    def to_msg_dataset(messages_df: pd.DataFrame) -> Dataset:
        """
        Convert the given DataFrame into a Dataset.

        Args:
            messages_df (pd.DataFrame): A DataFrame with columns ["group_id", "role", "content"].

        Returns:
            Dataset: A Dataset with a single column "messages" containing the messages.
        """
        messages = [
            group.to_dict("records") for _, group in messages_df.groupby("group_id")
        ]

        return Dataset.from_dict({"messages": messages})

    @staticmethod
    def _dialog_split(messages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO: Write tests to this method.
        """
        Split the given DataFrame into a list of DataFrames, each containing a part of a dialog and an answer DataFrame.

        Args:
            messages (pd.DataFrame): A DataFrame with columns ["group_id", "role", "content"].

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple of two DataFrames: the first one contains parts of dialogs and the
            second one contains answers.
        """
        logger = logging.getLogger(__name__)
        answers = []
        prompt_parts = []

        for group_id, group in messages.groupby("group_id"):
            formatted = [group[:i] for i in range(2, group.shape[0], 2)]
            # TODO: Move to enum, some role can have a different name.
            group_answer = group.loc[group["role"] == "assistant"][
                ["group_id", "content"]
            ]

            formatted = [df.astype(str) for df in formatted]
            for i, formatted_group in enumerate(formatted):
                formatted_group.loc[:, "group_id"] = f"{group_id}.{i}"

            if group_answer.shape[0] != len(formatted):
                logger.error(
                    "HfMsgDataset:prepare_to_train| Problems in provided dataset."
                    "Can't parse dialog for group_id: %s. This group will be skipped.",
                    group_id,
                )
                continue

            answers.append(group_answer)
            prompt_parts.extend(formatted)

        answers_df = pd.concat(answers)
        if answers_df.empty:
            logger.warning(
                "HfMsgDataset:prepare_to_train| Answers is empty. It can be a problem."
            )

        return pd.concat(prompt_parts), answers_df

    @staticmethod
    # TODO: Need to refactor this method. It work, but it was useful for train?
    #  It splits the dialog by subdialog, it okay but this method not worked for Casual Language Modeling.
    def prepare_to_train(
        messages_df: pd.DataFrame,
        configs: Optional[Config] = None,
        test_size: float = 0.1,
        subsplit: bool = False,
        tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None,
    ) -> tuple[Dataset, Dataset]:
        """
        Prepare a dataset for train by splitting it into train and test parts.

        This method splits the dialog by subdialog, it okay but this method not worked for Casual Language Modeling.

        Args:
            messages_df (pd.DataFrame): A DataFrame with columns ["group_id", "role", "content"].
            configs (Config, optional): Model configurations. Defaults to None.
            test_size (float, optional): Fraction of groups to include in the test set. Defaults to 0.1.
            subsplit (bool, optional): If True, then test will be splitted by subdialog. Defaults to False.
                Example of subsplit. If have dilog like that: sys/user/assistant/user/assistant.
                Then it will be splitted like: sys/usr + answer(assistant), sys/usr/assistant/usr + answer(assistant).
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast, optional): Tokenizer to use. Defaults to None.

        Returns:
            tuple[Dataset, Dataset]: A tuple of train and test datasets.
        """

        test_ds_size = round(test_size * messages_df["group_id"].unique().shape[0])
        test_ds_groups = messages_df["group_id"].sample(test_ds_size)

        test_df = messages_df.loc[messages_df["group_id"].isin(test_ds_groups)]
        train_df = messages_df.loc[~messages_df["group_id"].isin(test_ds_groups)]

        test_df, answers_df = HfMsgDataset._dialog_split(test_df)
        test_ds = HfMsgDataset(test_df, configs, tokenizer=tokenizer)

        if subsplit:
            test_ds.add_valid_data(answers_df["content"].to_list())

        train_ds = HfMsgDataset(train_df, configs, tokenizer)

        return train_ds.get_hf_dataset(), test_ds.get_hf_dataset(subsplit)

    @staticmethod
    def from_csv(csv_path: Path, configs: Config) -> "HfMsgDataset":
        """
        Create a HfMsgDataset from a CSV file.

        Args:
            csv_path (Path): The path to the CSV file.
            configs (Config): Model configurations.

        Returns:
            HfMsgDataset: The created dataset.
        """
        messages_df = pd.read_csv(csv_path)
        return HfMsgDataset(messages_df, configs)

    @staticmethod
    def from_df_one_sys(
        df: pd.DataFrame, configs: Config, sys_prompt: str
    ) -> "HfMsgDataset":
        """
        Create a HfMsgDataset from a pandas DataFrame with one system message for every group.

        Args:
            df (pd.DataFrame): A pandas DataFrame with columns ["role", "content", "group_id"].
            configs (Config): Model configurations.
            sys_prompt (str): System prompt to add to every group.

        Returns:
            HfMsgDataset: A HfMsgDataset with system prompts.
        """
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
            add_generation_prompt=self.configs.add_generation_prompt
            if self.configs is not None
            else False,
            tokenize=False,
        )  # type: ignore

        if group["role"].iat[-1] == "assistant":
            eos = self.tokenizer.eos_token

            end_index = prompt.rfind(eos)
            prompt = prompt[:end_index] + eos

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
        """
        Decode a batch of model output tokens into strings.

        Args:
            model_output_tokens (list[list[int]] | torch.Tensor): The model output tokens to decode.
            orig_cnt_tokens (int, optional): The original count of tokens to remove from the beginning of the decoded strings. Defaults to None.

        Returns:
            list[str]: The decoded strings.
        """
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
        """
        Decode a single model output token list into a string.

        Args:
            model_output_tokens (list[int]): The model output tokens to decode.

        Returns:
            str: The decoded string.
        """
        if self.tokenizer is None:
            raise ValueError("MsgDataset::decode| Tokenizer is None")

        return self.tokenizer.decode(model_output_tokens)

    @deprecated("Its method is not work properly at this moment.")
    def batch_encode_seq2seq(self, batch):
        # TODO: Check that it works with batch size > 1;

        model_input = self.tokenizer(
            batch["input_sequences"],
            padding=True,
            truncation=True,
            # Turn off, because sometimes it adds on more bos_token. We already have it  in `input_sequence`.
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=False,
        )

        labels = self.tokenizer(
            batch["target_sequence"],
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            add_special_tokens=False,
        )

        eos_id = self.tokenizer.eos_token_id
        skip_token = -100
        pad_id = self.tokenizer.pad_token_id
        cnt_ids = [len(input_ids) for input_ids in model_input["input_ids"]]

        model_input["input_ids"] = [
            input_ids + labels_ids + [eos_id]
            for input_ids, labels_ids in zip(
                model_input["input_ids"], labels["input_ids"]
            )
        ]

        model_input["attention_mask"] = [
            [1] * len(input_ids) for input_ids in model_input["input_ids"]
        ]

        labels["input_ids"] = [
            [skip_token] * cnt
            + [label if label != pad_id else skip_token for label in label_seq]
            + [eos_id]
            for label_seq, cnt in zip(labels["input_ids"], cnt_ids)
        ]

        model_input["labels"] = labels["input_ids"]
        assert len(model_input["labels"][0]) == len(model_input["input_ids"][0])
        return model_input

    def batch_encode(self, batch, add_valid=False) -> BatchEncoding:
        # TODO: Check that it works with batch size > 1;
        """
        Encode a batch of sequences using the Hugging Face tokenizer.

        Args:
            batch (dict): A batch of sequences with the following keys:
                - "input_sequences" (list[str]): The input sequences to encode.
                - "target_sequences" (list[str], optional): The target sequences to encode.
            add_valid (bool, optional): Whether to add the target sequences to the encoding.
                Defaults to False.

        Returns:
            BatchEncoding: The encoded batch.
        """
        model_input = self.tokenizer(
            batch["input_sequences"],
            text_target=batch["target_sequences"] if add_valid else None,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            # Turn off, because sometimes it adds one more bos_token. We already have it  in `input_sequence`.
            add_special_tokens=False,
        )

        return model_input

    def add_valid_data(self, valid_data: list[str]) -> None:
        """
        Add validation data to the dataset.

        Args:
            valid_data (list[str]): Validation data to add.
        """
        assert len(self.dataset) == len(valid_data)
        # TODO: Add merge by group_id
        for i, item in enumerate(valid_data):
            self.dataset[i].valid = item

    def get_hf_dataset(self, is_train=False) -> Dataset:
        """
        Return a Hugging Face dataset.

        Args:
            is_train (bool, optional): Whether the dataset is for training or not. Defaults to False.

        Returns:
            Dataset: Hugging Face dataset.
        """
        sentence = [item.sentence for item in self.dataset]
        groups_id = [item.group_id for item in self.dataset]
        valid = [item.valid for item in self.dataset]
        dataset = Dataset.from_dict(
            {
                "input_sequences": sentence,
                "target_sequences": valid,
                "group_id": groups_id,
            }
        )

        return dataset.map(
            lambda x: self.batch_encode(x, is_train),
            batched=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | slice) -> MsgDatasetItem | list[MsgDatasetItem]:
        return self.dataset[index]
