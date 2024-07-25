from enum import Enum


class RunnerType(Enum):
    HF = "hf"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    OPENAI = "openai"
