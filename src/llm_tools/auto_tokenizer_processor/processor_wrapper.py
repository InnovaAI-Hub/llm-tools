import logging

from transformers import (
    AutoProcessor,
    MllamaImageProcessor,
)
from typing import override
from llm_tools.config.config import Config
from .tokenizer_wrapper import TokenizerWrapper

logger = logging.getLogger(__name__)


class ProcessorWrapper(TokenizerWrapper):
    def __init__(
        self,
        config: Config,
        tokenizer: MllamaImageProcessor | None,
    ) -> None:
        super().__init__(config, tokenizer)

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
        tokenizer = AutoProcessor.from_pretrained(
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

    def encode(self, msg: str, image=None, **kwarg):
        max_seq_length = self.config.llm_model.max_seq_length
        tokens = self.tokenizer(
            msg, images=image, return_tensors="pt", max_length=max_seq_length
        )
        return tokens

    def batch_encode(self, batch: dict, add_valid=False, images=None, **kwarg):
        target = ""

        if add_valid and "target_sequences" in batch:
            target = batch["target_sequences"]

        tokens = self.tokenizer(
            batch["input_sequences"],
            text_target=target,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            # Turn off, because sometimes it adds one more bos_token. We already have it  in `input_sequence`.
            add_special_tokens=False,
        )

        return tokens
