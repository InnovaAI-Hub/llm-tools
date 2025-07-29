"""
Description:
    In this file we define the dataset class, class for using items in batch, and item class.
    Need a lot refactoring.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies:
    - pydantic
    - torch
    - transformers
    - pandas
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from datasets import Dataset
from llm_tools.auto_tokenizer_processor.abstract_wrapper import AbstractTokenizerWrapper
from llm_tools.config.config import Config
from llm_tools.config.dataset_config import DatasetConfig
from llm_tools.dataset.dialog import Dialog

logger = logging.getLogger(__name__)


# WARNING: This class is not fully supported yet, not tested and should not be used.
class AbstractDataset(ABC):
    def __init__(
        self,
        configs: DatasetConfig,
    ) -> None:
        self.configs = configs

    @abstractmethod
    def convert_to_hf(self, tokenizer: AbstractTokenizerWrapper) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def train_test_split(
        self, test_size: float = 0.1
    ) -> tuple["AbstractDataset", "AbstractDataset"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_from_csv(
        csv_path: Path, configs: Config, skip_tokenizer: bool = False
    ) -> "AbstractDataset":
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load_from_df_one_system(
        df: pd.DataFrame, configs: Config, sys_prompt: str
    ) -> "AbstractDataset":
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int | slice) -> Dialog | list[Dialog]:
        raise NotImplementedError
