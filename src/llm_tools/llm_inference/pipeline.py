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
        """
        Setup the pipeline by initializing the model runner based on the configuration.

        Raises:
            RuntimeError: id config is none.
        """

        if self.config is None:
            raise RuntimeError("Pipeline::_additional_setup| Config is not set")

        self.runner: AbstractModelRunner = RunnerGetter.get_runner(
            self.config.environment.runner_type, self.config
        )

    @override
    def _get_dataset(self, dataset_path: Path) -> tuple[Dataset, Dataset]:
        """
        Get the dataset from the dataset file and split it to train and test sets.

        Args:
            dataset_path (Path): Path to the dataset file. At this moment is only `csv` file.

        Raises:
            RuntimeError: if config is none.
            FileNotFoundError: if dataset file is not found.

        Returns:
            tuple[Dataset, Dataset]: Train and test datasets.

        TODO:
            Add support for `parquet` file.
        """
        if self.config is None:
            raise RuntimeError("Pipeline::_get_dataset| Config is not set")

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Pipeline::_get_dataset| Dataset file not found: {dataset_path}"
            )

        try:
            dataset_df = pd.read_csv(dataset_path)
            return HfMsgDataset.prepare_to_train(dataset_df, self.config)
        except Exception as e:
            raise RuntimeError(
                f"Pipeline::_get_dataset| Error while read or parse dataset: {e}"
            )

    @override
    def _run(
        self, dataset: Optional[HfMsgDataset | Dataset] = None
    ) -> list[ModelOutputItem]:
        """
        Run the pipeline on the dataset.

        Args:
            dataset (Optional[HfMsgDataset  |  Dataset], optional): Dataset to run the pipeline on. Defaults to None.
            TODO: Why is optional?

        Raises:
            ValueError: If dataset is None.

        Returns:
            list[ModelOutputItem]: List with model output items.
        """

        if dataset is None:
            raise ValueError("Pipeline::_run| Dataset is None")

        return self.runner.execute(dataset)
