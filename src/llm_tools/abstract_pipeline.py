# TODO: Update the logger msg.

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from datasets import Dataset
from llm_tools.config.config import Config
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from pydantic import BaseModel, ConfigDict, Field

from llm_tools.llm_inference.runner.abstract_model_runner import AbstractModelRunner


class AbstractPipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logger: logging.Logger = Field(default=logging.getLogger(__name__))
    config_path: Optional[Path] = Field(default=None)
    config: Optional[Config] = Field(default=None)
    runner: Optional[AbstractModelRunner] = Field(default=None)

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
        if self.config is not None:
            return self.config

        self.logger.debug("Pipeline::get_config| Config_path: %s", self.config_path)
        if self.config_path is not None:
            return Config.from_yaml(self.config_path)

        raise RuntimeError("Pipeline::get_config| Config is not set")

    @abstractmethod
    def _get_dataset(self, dataset_path: Path) -> HfMsgDataset | Dataset:
        raise NotImplementedError

    def get_dataset(self, dataset_path: Path) -> HfMsgDataset | Dataset:
        if self.config is None:
            raise RuntimeError("Pipeline::get_dataset| Config is not set")

        return self._get_dataset(dataset_path)

    @abstractmethod
    def _run(self, dataset: Optional[HfMsgDataset | Dataset] = None):
        raise NotImplementedError

    def run(self, dataset: HfMsgDataset | Dataset):
        self.logger.info("Pipeline::run| Start application.")
        try:
            if self.config is None:
                raise RuntimeError("Pipeline::run| Config is not set")

            return self._run(dataset)

        except Exception as error:
            self.logger.critical("MainApplication::run: %s", error, exc_info=True)
            return []
