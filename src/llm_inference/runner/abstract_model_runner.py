from abc import ABC, abstractmethod
import logging

from llm_inference.config.config import Config
from llm_inference.dataset.hf_msg_dataset import HfMsgDataset
from llm_inference.runner.model_output_item import ModelOutputItem


class AbstractModelRunner(ABC):
    def __init__(self, model_config: Config) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = model_config
        self.model_config = model_config.llm_model

    @abstractmethod
    def execute_once(self, input: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, input: HfMsgDataset) -> list[ModelOutputItem]:
        raise NotImplementedError
