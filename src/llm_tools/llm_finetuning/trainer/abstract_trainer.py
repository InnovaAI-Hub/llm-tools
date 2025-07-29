"""
Description:
    In this file we define the abstract trainer class.
    Need a lot refactoring.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 19.09.2024
Date Modified: 19.06.2024
Version: 0.1
Python Version: 3.12
Dependencies:
    - pydantic
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from datasets import Dataset as HFDataset
from llm_tools.auto_tokenizer_processor.abstract_wrapper import AbstractTokenizerWrapper
from llm_tools.config.dataset_config import DatasetConfig
from llm_tools.dataset.dataset import Dataset


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class PreparedDataset:
    train: HFDataset
    test: Optional[HFDataset]


class AbstractTrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, predict_labels):
        raise NotImplementedError

    @staticmethod
    def get_dataset(
        tokenizer: AbstractTokenizerWrapper,
        dataset_path: Path,
        config: DatasetConfig,
        eval_path: None | Path = None,
        test_size: float = 0.1,
    ) -> PreparedDataset:
        is_parquet = dataset_path.suffix == ".parquet"

        messages_df = (
            pd.read_parquet(dataset_path) if is_parquet else pd.read_csv(dataset_path)
        )

        dataset = Dataset(config)
        dataset.load_dataset(messages_df)

        res = None
        if eval_path is not None:
            is_parquet = eval_path.suffix == ".parquet"
            eval_df = (
                pd.read_parquet(eval_path) if is_parquet else pd.read_csv(eval_path)
            )

            test_ds = Dataset(config)
            test_ds.load_dataset(eval_df)

            res = PreparedDataset(
                train=dataset.convert_to_hf(tokenizer),
                test=test_ds.convert_to_hf(tokenizer),
            )

        else:
            train, test = dataset.train_test_split(test_size)
            res = PreparedDataset(
                train=train.convert_to_hf(tokenizer),
                test=test.convert_to_hf(tokenizer),
            )

        return res
