from enum import Enum

class MsgRoleType(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"