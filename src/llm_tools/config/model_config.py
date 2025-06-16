"""
Description: In this file we define the configuration for model and dataset.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pydantic, llm_tools.llm_inference
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

from llm_tools.type.model_dtype import ModelDType
from llm_tools.type.model_type import ModelType
from llm_tools.type.url_type import UrlType


class ModelConfigLLM(BaseSettings):
    llm_url: UrlType = Field(default="undefined", frozen=True)
    tokenizer_url: str | None = Field(
        default=None, frozen=True
    )  # TODO: Change to UrlType
    peft_path: Path | None = Field(default=None, frozen=True)
    llm_model_type: ModelType = Field(default=ModelType.LLAMA3, frozen=True)

    # Used when we load model from unsloth
    load_in_4bit: bool = Field(default=False, frozen=True)
    load_in_8bit: bool = Field(default=False, frozen=True)

    resize_embed_layer: int | None = Field(default=None, frozen=True)
    save_merged_model: bool = Field(default=False, frozen=True)

    token: str = Field(default="undefined", frozen=True)

    max_new_tokens: int = Field(default=512, frozen=True)  # for generation
    max_seq_length: int | None = Field(
        default=None, frozen=True
    )  # It used only in tokenize
    max_model_len: int | None = Field(
        default=None, frozen=True
    )  # It used for global tokenizer max limit

    temperature: float = Field(default=0.0, frozen=True, ge=0.0, le=1.0)
    top_p: float = Field(default=0.5, frozen=True, ge=0.0, le=1.0)

    terminators: list[int] = Field(default=[], frozen=True)
    pad_token_id: None | int = Field(default=None, frozen=True)
    pad_token: None | str = Field(default=None, frozen=True)

    do_sample: bool = Field(default=False, frozen=True)

    dtype: ModelDType = Field(default=ModelDType.BF16, frozen=True)
