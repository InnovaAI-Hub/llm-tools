from pathlib import Path
from llm_tools.config.model_config import ModelConfigLLM
from llm_tools.config.peft_method import PeftMethod
from pydantic import Field
from pydantic_settings import BaseSettings
from transformers import TrainingArguments


class ExperimentConfig(BaseSettings):
    experiment_name: str = Field("Undefined experiment", frozen=True)
    save_to: Path = Field(default=Path.cwd(), frozen=True)
    llm_model: ModelConfigLLM = Field(ModelConfigLLM(), frozen=True)
    peft_method: PeftMethod = Field(frozen=True)
    training_arguments: TrainingArguments = Field(frozen=True)
