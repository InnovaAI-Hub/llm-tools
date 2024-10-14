"""
Description:
    This file defines the AbstractModelRunner class, an abstract base class for running models
    with configurable inputs and outputs. It includes abstract methods for single and batch
    execution, which must be implemented by subclasses.

Classes:
    - AbstractModelRunner: An abstract base class for model execution with configurable inputs and outputs.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 03.10.2024
Version: 0.3
Python Version: 3.12
Dependencies:
"""

from abc import ABC, abstractmethod
import logging

from llm_tools.config.config import Config
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from llm_tools.llm_inference.runner.model_output_item import ModelOutputItem


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
