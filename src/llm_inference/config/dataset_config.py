"""
Description: In this file we define the configuration for the dataset.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pydantic
License: MIT License
"""

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    split_results_by: str | None = Field(default=None, frozen=True)
    add_generation_prompt: bool = Field(default=True, frozen=True)
    batch_count: int = Field(default=1, frozen=True, ge=1)
