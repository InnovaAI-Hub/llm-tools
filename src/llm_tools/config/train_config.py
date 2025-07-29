from pydantic_settings import BaseSettings
from pydantic import Field
from llm_tools.config.experiment_config import ExperimentConfig


class TrainConfig(BaseSettings):
    experiments: list[ExperimentConfig] = Field(default_factory=list)
