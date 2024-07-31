"""
Description: In this file we define the configuration for the dataset.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pydantic, RunnerType from llm_tools.llm_inference
"""

from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from llm_tools.type.runner_type import RunnerType


class EnvironmentConfig(BaseModel):
    runner_type: RunnerType = Field(default="hf", frozen=True)
    device_type: str = Field(default="auto", frozen=True)
    backup_path: Path = Field(default=Path("./backups"), frozen=True)
    num_workers: int = Field(default=0, frozen=True, ge=0)

    def model_post_init(self, __context: Any) -> None:
        if not self.backup_path.exists():
            self.backup_path.mkdir(parents=True)
