"""
Description:
    In this file we define the type of the runner.

Classes:
    - RunnerType: A class for storing the type of the runner.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 13.10.2024
Version: 0.1
Python Version: 3.10

WARNING: Llama cpp is not currently supported;
"""

from enum import StrEnum


class RunnerType(StrEnum):
    HF = "hf"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    OPENAI = "openai"
