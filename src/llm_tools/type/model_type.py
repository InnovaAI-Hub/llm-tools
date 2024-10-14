"""
Description:
    In this file we define the type of the model.

Classes:
    - ModelType: A class for storing the type of the model.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 13.10.2024
Version: 0.1
Python Version: 3.10
"""

from enum import StrEnum


class ModelType(StrEnum):
    LLAMA3 = "llama-3"
    # TODO: Add other model types
