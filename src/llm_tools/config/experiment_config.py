from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings
from transformers.training_args import TrainingArguments

from llm_tools.config.additional_token import AdditionalToken
from llm_tools.config.dataset_config import DatasetConfig
from llm_tools.config.model_config import ModelConfigLLM
from llm_tools.config.peft_method import PeftMethod
from llm_tools.config.metric_config import MetricConfig


class ExperimentConfig(BaseSettings):
    experiment_name: str = Field("Undefined experiment", frozen=True)
    peft_method: PeftMethod = Field(frozen=True)
    dataset_config: DatasetConfig = Field(frozen=True)
    save_to: Path = Field(default=Path.cwd(), frozen=True)
    save_merged: bool = Field(default=False, frozen=True)
    llm_model: ModelConfigLLM = Field(ModelConfigLLM(), frozen=True)
    training_arguments: TrainingArguments = Field(frozen=True)
    metric_config: MetricConfig = Field(frozen=True)
    additional_tokens: list[AdditionalToken] | None = Field(default=None, frozen=True)
    test_size: float = Field(default=0.1, frozen=True)
    eval_path: Path | None = Field(default=None, frozen=True)
