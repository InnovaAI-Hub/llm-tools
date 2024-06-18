from pydantic import BaseModel, Field
from llm_inference.type.runner_type import RunnerType


class GeneralConfig(BaseModel):
    runner_type: RunnerType = Field(default="hf", frozen=True)
    device_type: str = Field(default="auto", frozen=True)
