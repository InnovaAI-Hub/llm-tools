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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from llm_tools.config.config import Config
from pydantic import BaseModel, ConfigDict, Field, computed_field
from transformers.tokenization_utils_base import BatchEncoding


class MsgDatasetItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sentence: str = Field(default="")
    # Rename `valid`, it's `target` or `labels`
    valid: str = Field(default="")
    group_id: str | int = Field(default=0)

    tokens: Optional[BatchEncoding] = Field(default=None)

    @computed_field
    def cnt_tokens(self) -> int:
        return self.tokens["input_ids"].shape[1] if self.tokens is not None else 0  # type: ignore


# WARNING: This class is not fully supported yet, not tested and should not be used.
class AbstractMsgDataset(ABC):
    def __init__(
        self,
        messages_df: pd.DataFrame,
        configs: Optional[Config] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.configs = configs.dataset if configs is not None else None
        self.llm_model_type = (
            configs.llm_model.llm_model_type if configs is not None else None
        )

    @staticmethod
    @abstractmethod
    def from_csv(
        csv_path: Path, configs: Config, skip_tokenizer: bool = False
    ) -> "AbstractMsgDataset":
        raise NotImplementedError

    # @staticmethod
    # @abstractmethod
    # def get_formatter(model_type: ModelType) -> MsgFormatterFabric:
    #     raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_df_one_sys(
        df: pd.DataFrame, configs: Config, sys_prompt: str
    ) -> "AbstractMsgDataset":
        raise NotImplementedError

    @abstractmethod
    def format_dataset(self, messages_df: pd.DataFrame) -> list[MsgDatasetItem]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> MsgDatasetItem:
        raise NotImplementedError
