from typing import Any

import pandas as pd
from llm_inference.config.model_config import DatasetConfig
from llm_inference.dataset.msg_formatter import AbstractMsgFormatter, MsgFormatterFabric
from llm_inference.type.model_type import ModelType
from llm_inference.type.msg_role_type import MsgRoleType
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import BatchEncoding


# WARNING: This class is not fully supported yet, not tested and should not be used.
class MsgDataset(BaseModel):
    configs: DatasetConfig = Field(default=DatasetConfig(), frozen=True)
    messages_df: pd.DataFrame = Field(default=pd.DataFrame(None))
    sys_prompt: str = Field(default="", frozen=True, init=False)
    fmt: AbstractMsgFormatter = Field(frozen=True, init=False)
    llm_model_type: ModelType = Field(frozen=True)

    # Added because `Unable to generate pydantic-core schema for <class 'pandas.core.frame.DataFrame'>.``
    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        self.messages_df["role"] = self.messages_df["role"].apply(
            lambda x: MsgRoleType(x) # type: ignore
        )  # type: ignore
        self.sys_prompt = self._get_system_prompt()
        self.fmt = MsgFormatterFabric.get_formatter(self.llm_model_type)

    def _get_system_prompt(self) -> str:
        raise NotImplementedError

    def _format_messages(self):
        raise NotImplementedError

    def tokenize(self, tokenizer) -> list[BatchEncoding]:
        raise NotImplementedError

    def batch_decode(self, tokenizer, model_output_tokens) -> list[str]:
        raise NotImplementedError
