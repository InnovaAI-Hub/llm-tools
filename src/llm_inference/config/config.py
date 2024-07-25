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
from typing import Tuple, Type

from llm_inference.config.general_config import GeneralConfig
from llm_inference.config.model_config import ModelConfigLLM
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        yaml_file_encoding="utf-8",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    config_version: str = Field(default="1.0.0", frozen=True)
    general: GeneralConfig = Field(default=GeneralConfig(), frozen=True)
    llm_model: ModelConfigLLM = Field(default=ModelConfigLLM(token=""), frozen=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    @classmethod
    def from_yaml(cls, yaml_file: str | Path):
        cls.model_config["yaml_file"] = yaml_file
        return cls()
