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

import pandas as pd
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from datasets import Dataset


@dataclass(slots=True, config=ConfigDict(arbitrary_types_allowed=True))
class PreparedDataset:
    train: Dataset
    test: Dataset


class AbstractTrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, predict_labels):
        raise NotImplementedError

    @staticmethod
    def get_dataset(dataset_path: Path, tokenizer) -> PreparedDataset:
        is_parquet = dataset_path.suffix == ".parquet"

        messages_df = (
            pd.read_parquet(dataset_path) if is_parquet else pd.read_csv(dataset_path)
        )

        ds = HfMsgDataset.prepare_to_train(
            messages_df, tokenizer=tokenizer, test_size=0.1
        )

        return PreparedDataset(train=ds[0], test=ds[1])
