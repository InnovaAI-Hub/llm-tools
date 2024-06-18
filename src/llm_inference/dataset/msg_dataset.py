from typing import Any, Optional

import pandas as pd
import torch
from llm_inference.config.model_config import DatasetConfig
from llm_inference.dataset.msg_formatter import MsgFormatterFabric
from llm_inference.type.model_type import ModelType
from llm_inference.type.msg_role_type import MsgRoleType
from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import deprecated


class MsgDatasetItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    sentence: str = Field(default="")
    group_id: int = Field(default=0)
    device: str = Field(default="cuda")
    tokens: Optional[BatchEncoding] = Field(default=None)

    @computed_field
    def cnt_tokens(self) -> int:
        return self.tokens["input_ids"].shape[1] if self.tokens is not None else 0  # type: ignore


# TODO: Refactor this class
class MsgDatasetBatch:
    def __init__(
        self,
        batch: list[MsgDatasetItem],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> None:
        self.device = batch[0].device
        self.tokenizer = tokenizer
        self.batch_data: BatchEncoding = self._get_batch_tokens(batch)
        # NOTE: When we use data in batch, we need to apply padding.
        # Padding is added extra tokens at the end and at the beginning.
        # TODO: With padding we need only one count of tokens
        self.cnt_tokens_batch = self.batch_data.data["input_ids"].shape[1]
        self.len_seq_batch: list[int] = [len(item.sentence) for item in batch]
        self.groups_id = [item.group_id for item in batch]

    def _get_batch_tokens(self, batch: list[MsgDatasetItem]):
        # TODO: Check that we can use tokenizer.pad here
        # NOTE: It can be okay, but we tokenize twice. Need to refactor this.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        sequence: list[str] = [item.sentence for item in batch]
        return self.tokenizer(sequence, return_tensors="pt", padding=True)

    def to(self, device: str):
        self.device = device
        self.batch_data = self.batch_data.to(device)

    # TODO: Add support for pin_memory
    # def pin_memory(self):
    #     self.batch_data = {
    #         key: value.pin_memory(self.device) for key, value in self.batch_data.items()
    #     }
    #     return self


# WARNING: This class is not fully supported yet, not tested and should not be used.
class MsgDataset(Dataset):
    def __init__(
        self,
        messages_df: pd.DataFrame,
        llm_model_type: ModelType,
        configs: DatasetConfig,
        tokenizer_url: str,
        device: str = "cuda",
    ) -> None:
        self.tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(tokenizer_url, padding_side="left")
        )
        self.configs = configs
        self.llm_model_type = llm_model_type
        self.fmt = MsgFormatterFabric.get_formatter(self.llm_model_type)()
        self.dataset = self.format_dataset(messages_df)
        self.tokenize(self.tokenizer)

    def format_dataset(self, messages_df: pd.DataFrame) -> list[MsgDatasetItem]:
        # Apply transformations
        messages_df["role"] = messages_df["role"].apply(
            lambda x: MsgRoleType(x)  # type: ignore
        )

        if "group_id" not in messages_df.columns:
            messages_df["group_id"] = 0

        return [
            MsgDatasetItem(sentence=self.fmt.format_dialog(group), group_id=id)
            for id, group in messages_df.groupby("group_id")
        ]

    # NOTE: Maybe made this method static?
    def tokenize(self, tokenizer) -> None:
        for item in self.dataset:
            item.tokens = tokenizer(item.sentence, return_tensors="pt")

    def batch_decode(
        self,
        model_output_tokens: list[list[int]] | torch.Tensor,
        orig_cnt_tokens: Optional[int] = None,
    ) -> list[str]:
        batch = torch.tensor(model_output_tokens) if isinstance(model_output_tokens, list) else model_output_tokens
        batch = batch[:, orig_cnt_tokens:]
        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)


    def decode(self, model_output_tokens: list[int]) -> str:
        return self.tokenizer.decode(model_output_tokens)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> MsgDatasetItem:
        return self.dataset[index]
