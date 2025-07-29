"""
Description:
    In this file we define the data type of the model.

Classes:
    - ModelDType: A class for storing the data type of the model.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 18.06.2024
Date Modified: 13.10.2024
Version: 0.1
Python Version: 3.10

"""

from enum import StrEnum


class ModelDType(StrEnum):
    BF16 = "bfloat16"
    BF32 = "bfloat32"
    FP16 = "float16"
