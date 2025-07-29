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

WARNING: This class is not fully supported yet, not tested and should not be used. IT NOT REALIZED
"""

from pathlib import Path
from typing import Optional, override

from datasets import Dataset as HfMsgDataset
from llm_tools.abstract_pipeline import AbstractPipeline
from llm_tools.dataset.dataset import Dataset


class TrainerPipeline(AbstractPipeline):
    @override
    def _get_dataset(self, dataset_path: Path) -> tuple[Dataset, Dataset]:
        # dataset_df = pd.read_csv(dataset_path)
        raise NotImplementedError

    @override
    def _run(self, dataset: Optional[HfMsgDataset | Dataset] = None):
        raise NotImplementedError
