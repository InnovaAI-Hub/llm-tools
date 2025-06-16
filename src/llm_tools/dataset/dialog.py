from llm_tools.type.msg_role_type import MsgRoleType
from llm_tools.dataset.msg import BaseMessage, select_msg
from llm_tools.auto_tokenizer_processor.abstract_wrapper import AbstractTokenizerWrapper
import logging

logger = logging.getLogger(__name__)


class Dialog:
    def __init__(
        self,
    ) -> None:
        self.msgs: list[BaseMessage] = []
        self.group_id: str = ""
        self.last_role: MsgRoleType = MsgRoleType.SYSTEM
        self.used_roles: set[MsgRoleType] = set()

    def add_msg(self, msg: str, role: MsgRoleType) -> None:
        # 1. Check whether the role is system or not

        if role is MsgRoleType.SYSTEM:
            if role not in self.used_roles:
                self.used_roles.add(role)
                self.msgs.append(select_msg(role, msg))

            else:
                logger.warning(
                    "Dialog::add_msg: System msg already exists, ignore this one"
                )

        elif self.last_role != role:
            self.used_roles.add(role)
            self.msgs.append(select_msg(role, msg))
        else:
            self.msgs[-1].content += "\n" + msg

        self.last_role = role

    def add_image_msg(self, msg: str, role: MsgRoleType, image: bytes) -> None:
        if role is MsgRoleType.SYSTEM:
            raise ValueError("Cannot add image msg to system role")

        if self.last_role != role:
            self.used_roles.add(role)
            self.msgs.append(select_msg(role, msg, image))
        else:
            self.msgs[-1].content += "\n" + msg

        self.last_role = role

    def format_dialog(self, tokenizer: AbstractTokenizerWrapper) -> str:
        dialog = [item.to_dict() for item in self.msgs]
        result: str = tokenizer.apply_chat_template(dialog)
        return result
