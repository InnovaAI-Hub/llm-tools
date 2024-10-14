"""
Description:
    This file defines the TrainerPipeline class, which loads and prepares datasets
    for language model training by reading a CSV and processing it with Hugging Face's tools.

Classes:
    - TrainerPipeline: Extends AbstractPipeline to fetch and prepare datasets for training.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 03.10.2024
Version: 0.3
Python Version: 3.12
Dependencies:
    - pandas
    - datasets

WARNING: This class is not fully supported yet, not tested and should not be used.
"""

from pathlib import Path
from typing import Optional, override

import pandas as pd
from datasets import Dataset
from llm_tools.abstract_pipeline import AbstractPipeline
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset


class TrainerPipeline(AbstractPipeline):
    @override
    def _get_dataset(self, dataset_path: Path) -> HfMsgDataset | Dataset:
        dataset_df = pd.read_csv(dataset_path)
        return HfMsgDataset.prepare_to_train(dataset_df, self.config)

    @override
    def _run(self, dataset: Optional[HfMsgDataset | Dataset] = None):
        raise NotImplementedError
