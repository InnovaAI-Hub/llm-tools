import logging

import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing_extensions import override

from llm_tools.config.model_config import ModelConfigLLM

from .abstract_wrapper import AbstractTokenizerWrapper

logger = logging.getLogger(__name__)


class TokenizerWrapper(AbstractTokenizerWrapper):
    def __init__(
        self,
        config: ModelConfigLLM,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None,
    ) -> None:
        super().__init__(config)

        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            self._check_tokenizer_initialization(config, tokenizer)
        )

    def _check_tokenizer_initialization(
        self,
        llm_model_conf: ModelConfigLLM | None,
        tokenizer: None | PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        This method checks if the tokenizer is already initialized and if so, returns it.
        If the tokenizer is not initialized, it will try to initialize it with the given
        `llm_model_conf` and `tokenizer` parameters.

        Args:
            llm_model_conf (Optional[ModelConfigLLM]): The ModelConfigLLM object with settings for the model.
            tokenizer (Optional[PreTrainedTokenizer|PreTrainedTokenizerFast]): The tokenizer for the model.

        Raises:
            ValueError: If both `llm_model_conf` and `tokenizer` are None.

        Returns:
            PreTrainedTokenizer | PreTrainedTokenizerFast: The initialized tokenizer.
        """
        if tokenizer is not None:
            return tokenizer

        if llm_model_conf is None:
            raise ValueError(
                "HfMsgDataset::init_tokenizer| `llm_model_conf` is None and `tokenizer` is None"
            )

        # Select tokenizer path
        tokenizer_path = (
            llm_model_conf.llm_url
            if llm_model_conf.tokenizer_url is None
            else llm_model_conf.tokenizer_url
        )

        logger.debug(
            "HfMsgDataset::init_tokenizer| "
            f"Initializing tokenizer from {tokenizer_path}."
        )

        return self.get_tokenizer(
            tokenizer_path,
            pad_token_id=llm_model_conf.pad_token_id,
            pad_token=llm_model_conf.pad_token,
            max_length=llm_model_conf.max_model_len,
        )

    @override
    @staticmethod
    def setup_tokenizer(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        set_pad_token: bool = True,
        pad_token: None | str = None,
        pad_token_id: None | int = None,
        max_length: None | int = None,
    ) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
        """
        Setup the tokenizer with the given parameters.

        Args:
            tokenizer (PreTrainedTokenizer|PreTrainedTokenizerFast): The tokenizer to setup.
            set_pad_token (bool, optional): Whether to set the pad token. Defaults to True.
            pad_token (str, optional): The pad token. Defaults to None.
            pad_token_id (int, optional): The pad token id. Defaults to None.
            max_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            PreTrainedTokenizer|PreTrainedTokenizerFast: The setup tokenizer.
        """
        logger = logging.getLogger(__name__)

        if max_length is not None:
            tokenizer.model_max_length = max_length

        if set_pad_token and not tokenizer.pad_token:
            logger.debug(
                "HfMsgDataset::get_tokenizer| Set pad token."
                f"Current pad_token: ({tokenizer.pad_token}, {tokenizer.pad_token_id}),"
                f"Eos_token: ({tokenizer.eos_token}, {tokenizer.eos_token_id})."
            )

            tokenizer.pad_token = (
                pad_token if pad_token is not None else tokenizer.eos_token
            )

            tokenizer.pad_token_id = (
                pad_token_id if pad_token_id is not None else tokenizer.eos_token_id
            )

        logger.debug(
            "HfMsgDataset::get_tokenizer| "
            f"Current pad_token: ({tokenizer.pad_token}, {tokenizer.pad_token_id}),"
            f"Eos_token: ({tokenizer.eos_token}, {tokenizer.eos_token_id})."
        )

        return tokenizer

    @override
    @staticmethod
    def get_tokenizer(
        url: str,
        set_pad_token: bool = True,
        pad_token: None | str = None,
        pad_token_id: None | int = None,
        max_length: None | int = None,
    ):
        """
        Get the tokenizer from the given `url`.

        Args:
            url (str): The URL to the tokenizer.
            set_pad_token (bool, optional): Whether to set the pad token. Defaults to True.
            pad_token (str, optional): The pad token. Defaults to None.
            pad_token_id (int, optional): The pad token id. Defaults to None.
            max_length (int, optional): The maximum sequence length. Defaults to None.

        Returns:
            PreTrainedTokenizer|PreTrainedTokenizerFast: The setup tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            url,
            padding_side="left",
            use_fast=True,
        )

        tokenizer = TokenizerWrapper.setup_tokenizer(
            tokenizer=tokenizer,
            set_pad_token=set_pad_token,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            max_length=max_length,
        )

        return tokenizer

    @override
    def batch_decode(
        self,
        model_output_tokens: list[list[int]] | torch.Tensor,
        orig_cnt_tokens: None | int = None,
    ) -> list[str]:
        """
        Decode a batch of model output tokens into strings.

        Args:
            model_output_tokens (list[list[int]] | torch.Tensor): The model output tokens to decode.
            orig_cnt_tokens (int, optional): The original count of tokens to remove from the beginning of the decoded strings. Defaults to None.

        Returns:
            list[str]: The decoded strings.
        """
        if self.tokenizer is None:
            raise ValueError("MsgDataset::batch_decode| Tokenizer is None")

        batch = (
            torch.tensor(model_output_tokens)
            if isinstance(model_output_tokens, list)
            else model_output_tokens
        )
        batch = batch[:, orig_cnt_tokens:]
        result = self.tokenizer.batch_decode(batch, skip_special_tokens=True)
        return result

    @override
    def decode(self, model_output_tokens: list[int]) -> str:
        """
        Decode a single model output token list into a string.

        Args:
            model_output_tokens (list[int]): The model output tokens to decode.

        Returns:
            str: The decoded string.
        """
        if self.tokenizer is None:
            raise ValueError("MsgDataset::decode| Tokenizer is None")

        result = self.tokenizer.decode(model_output_tokens)
        return result

    @override
    def encode(self, msg: str, **kwarg):
        max_seq_length = self.config.max_seq_length
        tokens = self.tokenizer(msg, return_tensors="pt", max_length=max_seq_length)
        return tokens

    @override
    def apply_chat_template(
        self, msg: list[dict[str, str]], add_generation_prompt=True
    ) -> str:
        result = self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=add_generation_prompt
        )

        if not isinstance(result, str):
            raise ValueError(f"tokenizer.apply_chat_template: {type(result)}")

        return result

    @override
    def batch_encode(self, batch: dict, add_valid=False, **kwarg):
        target = None

        if add_valid and "target_sequences" in batch:
            target = batch["target_sequences"]

        tokens = self.tokenizer(
            batch["input_sequences"],
            text_target=target,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            # Turn off, because sometimes it adds one more bos_token. We already have it  in `input_sequence`.
            add_special_tokens=True,
        )

        return tokens

    @override
    def get_current_tokenizer(self):
        return self.tokenizer
