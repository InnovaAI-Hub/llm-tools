"""
Description: In this file we define the configuration for model and dataset.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pydantic, llm_inference
"""

from llm_inference.type.model_type import ModelType
from llm_inference.type.url_type import UrlType
from llm_inference.type.model_dtype import ModelDType
from llm_inference.config.dataset_config import DatasetConfig
from pydantic import BaseModel, Field


class ModelConfigLLM(BaseModel):
    llm_url: UrlType = Field(default="", frozen=True)
    llm_model_type: ModelType = Field(default=ModelType.LLAMA3, frozen=True)
    token: str = Field(default="undefined", frozen=True)

    max_new_tokens: int = Field(default=512, frozen=True)

    temperature: float = Field(default=0.0, frozen=True, ge=0.0, le=1.0)
    terminators: list[int] = Field(default=[], frozen=True)
    top_p: float = Field(default=0.0, frozen=True, ge=0.0, le=1.0)
    pad_token_id: int = Field(default=0, frozen=True)

    do_sample: bool = Field(default=False, frozen=True)

    dataset: DatasetConfig = Field(default=DatasetConfig(), frozen=True)

    dtype: ModelDType = Field(default=ModelDType.BF16, frozen=True)
