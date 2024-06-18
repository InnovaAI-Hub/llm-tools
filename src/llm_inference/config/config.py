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
        yaml_file="config.yaml", yaml_file_encoding="utf-8"
    )

    config_version: str = Field(default="1.0.0", frozen=True)
    general: GeneralConfig = Field(default=GeneralConfig(), frozen=True)
    llm_model: ModelConfigLLM = Field(default=ModelConfigLLM(), frozen=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)

    @classmethod
    def from_yaml(cls, yaml_file: str | Path):
        cls.model_config["yaml_file"] = yaml_file
        return cls()
