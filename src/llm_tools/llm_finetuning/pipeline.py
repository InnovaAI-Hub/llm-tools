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
