from abc import abstractmethod, ABC
from io import BytesIO
from typing_extensions import override
from pydantic import BaseModel

from llm_tools.type.msg_role_type import MsgRoleType
from llm_tools.type.model_type import ModelType
from PIL import Image


class AbstractMessage(BaseModel, ABC):
    role: MsgRoleType
    content: str
    image: bytes | None = None

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class BaseMessage(AbstractMessage):
    @override
    def to_dict(self) -> dict:
        result: dict[str, str] = {
            "role": self.role.name.lower(),
            "content": self.content,
        }

        return result


class Gemma3Message(AbstractMessage):
    @override
    def to_dict(self) -> dict[str, str | list[dict[str, str]]]:
        result: dict[str, str | list[dict[str, str]]] = {
            "role": self.role.name.lower(),
            "content": [
                {"type": "text", "text": self.content},
            ],
        }

        if self.image:
            tmp: str | list[dict[str, str]] = result["content"]
            if isinstance(tmp, list):
                tmp.insert(0, {"type": "image"})
            result["content"] = tmp

        return result


class Llama3VisionMessage(AbstractMessage):
    @override
    def to_dict(self) -> dict[str, str | list[dict[str, str]]]:
        result: dict[str, str | list[dict[str, str]]] = {
            "role": self.role.name.lower(),
            "content": [
                {"type": "text", "text": self.content},
            ],
        }

        if self.image:
            tmp: str | list[dict[str, str]] = result["content"]
            if isinstance(tmp, list):
                tmp.insert(0, {"type": "image"})
            result["content"] = tmp

        return result


def to_pil_image(image: bytes) -> Image.Image:
    io_stream = BytesIO(image)
    pil_image = Image.open(io_stream)
    return pil_image


def select_msg(
    role: MsgRoleType,
    content: str,
    image: bytes | None = None,
    model_type: ModelType | None = None,
) -> BaseMessage:
    msgs = {
        None: BaseMessage,
        ModelType.GEMMA3: Gemma3Message,
        ModelType.LLAMA3: Llama3VisionMessage,
    }

    result = msgs[model_type]
    return result(role=role, content=content, image=image)
