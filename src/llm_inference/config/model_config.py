from llm_inference.type.url_type import UrlType
from llm_inference.type.model_dtype import ModelDType
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    split_results_by: str | None = Field(default=None, frozen=True)
    add_generation_prompt: bool = Field(default=True, frozen=True)


class ModelConfigLLM(BaseModel):
    llm_url: UrlType = Field(default="", frozen=True)
    max_new_tokens: int = Field(default=512, frozen=True)

    temperature: float = Field(default=0.0, frozen=True, ge=0.0, le=1.0)
    terminators: list[int] = Field(default=[], frozen=True)
    top_p: float = Field(default=0.0, frozen=True, ge=0.0, le=1.0)
    pad_token_id: int = Field(default=0, frozen=True)
    
    do_sample: bool = Field(default=False, frozen=True)

    dataset: DatasetConfig = Field(default=DatasetConfig(), frozen=True)

    dtype: ModelDType = Field(default=ModelDType.BF16, frozen=True)