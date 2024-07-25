from enum import Enum


class ModelDType(Enum):
    BF16: str = "bfloat16"
    BF32: str = "bfloat32"
    FP16: str = "float16"
