from abc import ABC, abstractmethod

import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from llm_tools.config.model_config import ModelConfigLLM


class AbstractTokenizerWrapper(ABC):
    def __init__(self, config: ModelConfigLLM):
        self.config = config

    @staticmethod
    @abstractmethod
    def setup_tokenizer(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        set_pad_token: bool = True,
        pad_token: None | str = None,
        pad_token_id: None | int = None,
        max_length: None | int = None,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_tokenizer(
        url: str,
        set_pad_token: bool = True,
        pad_token: None | str = None,
        pad_token_id: None | int = None,
        max_length: None | int = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def get_current_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def batch_decode(
        self,
        model_output_tokens: list[list[int]] | torch.Tensor,
        orig_cnt_tokens: None | int = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, model_output_tokens: list[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, msg: str, **kwarg):
        raise NotImplementedError

    @abstractmethod
    def batch_encode(self, batch: dict, add_valid=False, **kwarg):
        raise NotImplementedError

    @abstractmethod
    def apply_chat_template(self, msg: list[dict[str, str]]) -> str:
        raise NotImplementedError
