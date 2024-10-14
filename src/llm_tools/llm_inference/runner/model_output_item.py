"""
Description:
    In this file we define the model output item class.

Classes:
    - ModelOutputItem: A class for storing model output items.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 13.10.2024
Version: 0.1
Python Version: 3.10
Dependencies:
    - pydantic
"""

from pydantic import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class ModelOutputItem:
    group_id: int | str
    text: str
