from abc import ABC, abstractmethod
import logging

from llm_inference.config.config import Config


class AbstractModelRunner(ABC):
    def __init__(self, model_config: Config) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = model_config
        self.model_config = model_config.llm_model

    @abstractmethod
    def execute(self, input: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute_batch(self, input: list[str]) -> list[str]:
        raise NotImplementedError
