"""
Description: In this file we define the configuration for the dataset.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pydantic
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class DatasetConfig(BaseSettings):
    add_generation_prompt: bool = Field(default=True)
    batch_size: int = Field(default=1, ge=1)
