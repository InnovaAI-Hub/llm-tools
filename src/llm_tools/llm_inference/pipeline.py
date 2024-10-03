from pathlib import Path
from typing import Optional, override

import pandas as pd
from datasets import Dataset
from llm_tools.abstract_pipeline import AbstractPipeline
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from llm_tools.llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_tools.llm_inference.runner.model_output_item import ModelOutputItem
from llm_tools.llm_inference.runner.runner_getter import RunnerGetter


class Pipeline(AbstractPipeline):
    @override
    def _additional_setup(self) -> None:
        if self.config is None:
            raise RuntimeError("Pipeline::_additional_setup| Config is not set")

        self.runner: AbstractModelRunner = RunnerGetter.get_runner(
            self.config.environment.runner_type, self.config
        )

    @override
    def _get_dataset(self, dataset_path: Path) -> HfMsgDataset | Dataset:
        if self.config is None:
            raise RuntimeError("Pipeline::_get_dataset| Config is not set")

        dataset_df = pd.read_csv(dataset_path)
        return HfMsgDataset.prepare_to_train(dataset_df, self.config)

    @override
    def _run(
        self, dataset: Optional[HfMsgDataset | Dataset] = None
    ) -> list[ModelOutputItem]:
        return self.runner.execute(dataset)
