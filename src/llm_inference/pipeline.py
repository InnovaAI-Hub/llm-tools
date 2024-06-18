import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import Field, BaseModel, ConfigDict

from llm_inference.config.config import Config
from llm_inference.dataset.msg_dataset import MsgDataset
from llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_inference.runner.runner_getter import RunnerGetter
from llm_inference.type.model_type import ModelType


class Pipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    logger: logging.Logger = Field(default=logging.getLogger(__name__))
    config_path: Optional[Path]
    config: Optional[Config] = None
    results: Optional[list[dict]] = Field(default=None, init=False)

    def model_post_init(self, __context) -> None:
        self.config = self.get_config()

    def get_config(self) -> Config:
        if self.config is not None:
            return self.config

        self.logger.debug("Pipeline::get_config| Config_path: %s", self.config_path)
        if self.config_path is not None:
            return Config.from_yaml(self.config_path)

        raise RuntimeError("Pipeline::get_config| Config is not set")

    def get_dataset(self, dataset_path: Path) -> MsgDataset:
        if self.config is None:
            raise RuntimeError("Pipeline::get_dataset| Config is not set")

        return MsgDataset(
            pd.read_csv(dataset_path),
            ModelType.LLAMA3,
            self.config.llm_model.dataset,
            self.config.llm_model.llm_url,
        )

    def run(self, dataset_path: Path):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self.logger.info("MainApplication::run| Start application.")
        try:
            if self.config is None:
                raise RuntimeError("Pipeline::run| Config is not set")

            dataset = self.get_dataset(dataset_path)
            # TODO: Maybe move config var to get_runner?
            runner: AbstractModelRunner = RunnerGetter.get_runner(
                self.config.general.runner_type
            )(self.config)

            return runner.execute(dataset)

        except Exception as error:
            self.logger.critical("MainApplication::run: %s", error, exc_info=True)
            return []
