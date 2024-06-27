"""
Description: Module for formatting messages. Maybe need switch to Jinja or similar.
WARNING: This class is not fully supported yet, not tested and should not be used.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: pandas
"""

import logging
from abc import ABC, abstractmethod

import pandas as pd
from llm_inference.type.model_type import ModelType
from llm_inference.type.msg_role_type import MsgRoleType


class AbstractMsgFormatter(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def is_correct_msg_df(self, msg_df: pd.DataFrame):
        # Need that first string - is system prompt
        # Then pairs with user and assistant msg. First user msg than assistant.
        for _, group in msg_df.groupby("group_id"):
            msg_tmp: pd.DataFrame = group.reset_index()
            user_role_index = [x for x in range(1, msg_tmp.shape[0], 2)]
            assist_role_index = [x for x in range(2, msg_tmp.shape[0], 2)]

            is_correct_user_index: bool = (
                msg_tmp["role"].iloc[user_role_index] == MsgRoleType.USER  # type: ignore
            ).all()  # type: ignore
            is_correct_assist_index: bool = (
                msg_tmp["role"].iloc[assist_role_index] == MsgRoleType.ASSISTANT  # type: ignore
            ).all()  # type: ignore

            if not (is_correct_user_index and is_correct_assist_index):
                raise ValueError(
                    f"MsgFormatter:is_correct_msg_df| Wrong order of roles. Right: system/user/assistant..."
                    f"Is user index correct: {is_correct_user_index}, is assist index correct: {is_correct_assist_index}",
                )

            # Check that last msg by user
            if msg_tmp["role"].iloc[-1] != MsgRoleType.USER:
                raise ValueError(
                    "MsgFormatter:is_correct_msg_df| Last msg is not by user"
                )

    @abstractmethod
    def format_msg(self, content: str, role: MsgRoleType) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_prompt_start(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_prompt_end(self) -> str:
        raise NotImplementedError

    def format_dialog(self, msg_df: pd.DataFrame) -> str:
        self.is_correct_msg_df(msg_df)

        formatted_msg: pd.Series[str] = msg_df.apply(
            lambda x: self.format_msg(x["content"], x["role"]), axis=1
        )

        return f"{self.get_prompt_start()}{formatted_msg.sum()}{self.get_prompt_end()}"


class LLAMA3Formatter(AbstractMsgFormatter):
    def get_prompt_start(self) -> str:
        return "<|begin_of_text|>"

    def format_msg(self, content: str, role: MsgRoleType) -> str:
        return (
            f"<|start_header_id|>{role.value}<|end_header_id|>\n\n{content}<|eot_id|>"
        )

    def get_prompt_end(self) -> str:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"


class MsgFormatterFabric:
    @staticmethod
    def get_formatter(model_name: ModelType) -> type[AbstractMsgFormatter]:
        formatters = {
            ModelType.LLAMA3: LLAMA3Formatter,
        }

        return formatters.get(model_name)  # type: ignore
