from pathlib import Path

import numpy as np
import pandas as pd
from typing_extensions import override

from datasets import Dataset as HFDataset
from llm_tools.auto_tokenizer_processor.abstract_wrapper import AbstractTokenizerWrapper
from llm_tools.config.config import Config
from llm_tools.config.dataset_config import DatasetConfig
from llm_tools.dataset.abstract_dataset import AbstractDataset
from llm_tools.dataset.dialog import Dialog
from llm_tools.type.msg_role_type import MsgRoleType


class Dataset(AbstractDataset):
    def __init__(
        self,
        config: DatasetConfig,
    ):
        super().__init__(config)
        self.dialogs: list[Dialog] = []

    def load_dataset(self, df: pd.DataFrame):
        dialogs = []
        for group_id, group in df.groupby("group_id"):
            dialog = Dialog()
            for row in group.itertuples():
                msg = str(row.content)
                role = MsgRoleType(row.role)
                dialog.add_msg(msg, role)

            dialog.group_id = str(group_id)
            dialogs.append(dialog)

        self.dialogs = dialogs

    def load_from_parquet(self, dataset_path: Path):
        df: pd.DataFrame = pd.read_parquet(dataset_path)
        self.load_dataset(df)

    def convert_to_hf(self, tokenizer: AbstractTokenizerWrapper) -> HFDataset:
        sentence = [item.format_dialog(tokenizer) for item in self.dialogs]
        groups_id = [item.group_id for item in self.dialogs]
        dataset = HFDataset.from_dict(
            {
                "input_sequences": sentence,
                "group_id": groups_id,
            }
        )

        return dataset.map(
            lambda x: tokenizer.batch_encode(x),
            batched=True,
        )

    @override
    def train_test_split(self, test_size: float = 0.1) -> tuple["Dataset", "Dataset"]:
        rng = np.random.default_rng(seed=42)
        size: int = round(len(self) * test_size)
        test_index = rng.choice(len(self), size)

        train: list[Dialog] = [
            item for i, item in enumerate(self.dialogs) if i not in test_index
        ]
        test: list[Dialog] = [self.dialogs[i] for i in test_index]

        train_ds = Dataset(self.configs)
        train_ds.dialogs = train

        test_ds = Dataset(self.configs)
        test_ds.dialogs = test

        return train_ds, test_ds

    @override
    @staticmethod
    def load_from_csv(csv_path: Path, configs: Config, skip_tokenizer: bool = False):
        raise NotImplementedError

    @override
    @staticmethod
    def load_from_df_one_system(df: pd.DataFrame, configs: Config, sys_prompt: str):
        raise NotImplementedError

    @override
    @override
    def __len__(self) -> int:
        return len(self.dialogs)

    @override
    def __getitem__(self, index: int | slice) -> Dialog | list[Dialog]:
        return self.dialogs[index]
