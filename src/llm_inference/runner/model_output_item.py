from pydantic import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class ModelOutputItem:
    group_id: int | str
    text: str
