from pydantic_settings import BaseSettings
from peft import LoraConfig


class PeftMethod(BaseSettings):
    name: str
    lora_conf: LoraConfig
