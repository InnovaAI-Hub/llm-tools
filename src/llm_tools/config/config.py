"""
Description: In this file we define the main config files.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pydantic
"""

from pathlib import Path

from llm_tools.config.dataset_config import DatasetConfig
from llm_tools.config.environment_config import EnvironmentConfig
from llm_tools.config.model_config import ModelConfigLLM
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from llm_tools.config.train_config import TrainConfig


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        validate_assignment=True,
        validate_default=False,
    )

    config_version: str = Field(default="0.4.7", frozen=True)
    environment: EnvironmentConfig = Field(default=EnvironmentConfig(), frozen=True)
    llm_model: ModelConfigLLM = Field(default=ModelConfigLLM(token=""), frozen=True)
    dataset: DatasetConfig = Field(default=DatasetConfig(), frozen=True)
    train: TrainConfig | None = Field(default=None, frozen=True)

    @field_validator("config_version")
    def validate_config_version(cls, config_version: str) -> str:
        if config_version != "0.4.7":
            raise ValueError(
                f"Config version {config_version} is not supported. Check current version in docs."
            )
        return config_version

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_yaml(cls, yaml_file: str | Path):
        assert Path(yaml_file).exists(), "Check that file exists."

        cls.model_config["yaml_file"] = yaml_file
        return cls()
