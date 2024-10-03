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

from typing import Optional
from pydantic_settings import BaseSettings
from llm_tools.type.model_type import ModelType
from llm_tools.type.url_type import UrlType
from llm_tools.type.model_dtype import ModelDType
from pydantic import Field


class ModelConfigLLM(BaseSettings):
    llm_url: UrlType = Field(default="undefined", frozen=True)
    tokenizer_url: Optional[str] = Field(default=None, frozen=True)
    peft_path: Optional[str] = Field(default=None, frozen=True)
    llm_model_type: ModelType = Field(default=ModelType.LLAMA3, frozen=True)

    token: str = Field(default="undefined", frozen=True)

    max_new_tokens: int = Field(default=512, frozen=True)  # for generation
    max_tokens: Optional[int] = Field(default=None, frozen=True)  # for tokenization
    max_seq_length: Optional[int] = Field(default=None, frozen=True)
    max_model_len: Optional[int] = Field(default=None, frozen=True)

    temperature: float = Field(default=0.0, frozen=True, ge=0.0, le=1.0)
    top_p: float = Field(default=0.5, frozen=True, ge=0.0, le=1.0)

    terminators: list[int] = Field(default=[], frozen=True)
    pad_token_id: Optional[int] = Field(default=None, frozen=True)
    pad_token: Optional[str] = Field(default=None, frozen=True)

    do_sample: bool = Field(default=False, frozen=True)

    dtype: ModelDType = Field(default=ModelDType.BF16, frozen=True)
