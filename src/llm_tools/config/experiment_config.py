from pathlib import Path
from typing import Optional

from llm_tools.config.additional_token import AdditionalToken
from llm_tools.config.model_config import ModelConfigLLM
from llm_tools.config.peft_method import PeftMethod
from pydantic import Field
from pydantic_settings import BaseSettings
from transformers import TrainingArguments


class ExperimentConfig(BaseSettings):
    experiment_name: str = Field("Undefined experiment", frozen=True)
    peft_method: PeftMethod = Field(frozen=True)
    save_to: Path = Field(default=Path.cwd(), frozen=True)
    save_merged: bool = Field(default=False, frozen=True)
    llm_model: ModelConfigLLM = Field(ModelConfigLLM(), frozen=True)
    training_arguments: TrainingArguments = Field(frozen=True)
    metric: str = Field(default="exact_match", frozen=True)
    additional_tokens: Optional[list[AdditionalToken]] = Field(
        default=None, frozen=True
    )
