"""
Description:
    Abstract base class for pipelines.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 13.10.2024
Version: 0.1
Python Version: 3.10
Dependencies:
    - pydantic
    - datasets (hf)

"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from datasets import Dataset
from llm_tools.config.config import Config
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from pydantic import BaseModel, ConfigDict, Field

from llm_tools.llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_tools.llm_inference.runner.model_output_item import ModelOutputItem
from llm_tools.llm_finetuning.trainer.abstract_trainer import AbstractTrainer


class AbstractPipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logger: logging.Logger = Field(default=logging.getLogger(__name__))
    config_path: Optional[Path] = Field(default=None)
    config: Optional[Config] = Field(default=None)
    runner: Optional[AbstractModelRunner] = Field(default=None)
    trainer: Optional[AbstractTrainer] = Field(default=None)

    def model_post_init(self, __context) -> None:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self.config = self.get_config()
        self._additional_setup()

    @abstractmethod
    def _additional_setup(self) -> None:
        raise NotImplementedError

    def get_config(self) -> Config:
        """
        Gets the config for this pipeline.
        If the config is already set, it will be returned.
        Otherwise, if a config path is set, it will be used to load the config from a yaml file.

        Raises:
            RuntimeError: If no config path is set.

        Returns:
            Config: The config for this pipeline.
        """
        if self.config is not None:
            return self.config

        self.logger.debug("Pipeline::get_config| Config_path: %s", self.config_path)
        if self.config_path is not None:
            return Config.from_yaml(self.config_path)

        raise RuntimeError("Pipeline::get_config| Config is not set")

    @abstractmethod
    def _get_dataset(self, dataset_path: Path) -> tuple[Dataset, Dataset]:
        raise NotImplementedError

    def get_dataset(self, dataset_path: Path) -> tuple[Dataset, Dataset]:
        """
        Gets the dataset for this pipeline.

        Args:
            dataset_path (Path): The path to the dataset.

        Raises:
            RuntimeError: If the config is not set.

        Returns:
            HfMsgDataset | Dataset:
        """
        if self.config is None:
            raise RuntimeError("Pipeline::get_dataset| Config is not set")

        return self._get_dataset(dataset_path)

    @abstractmethod
    def _run(
        self, dataset: Optional[HfMsgDataset | Dataset] = None
    ) -> list[ModelOutputItem]:
        raise NotImplementedError

    def run(self, dataset: HfMsgDataset | Dataset) -> list[ModelOutputItem]:
        """
        Runs the pipeline on the dataset.

        Args:
            dataset (HfMsgDataset | Dataset): The dataset to run the pipeline on.

        Raises:
            RuntimeError: If the config is not set.

        Returns:
            list[ModelOutputItem]: List with model output items.
        """
        self.logger.info("Pipeline::run| Start application.")
        try:
            if self.config is None:
                raise RuntimeError("Pipeline::run| Config is not set")

            return self._run(dataset)

        except Exception as error:
            self.logger.critical("Pipeline::run: %s", error, exc_info=True)
            return []
