import logging
from pathlib import Path

from llm_tools.config.config import Config
from llm_tools.llm_finetuning.trainer.unsloth_trainer import UnslothTrainer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_dataset_pth() -> Path:
    return Path("formatted_train_dataset.parquet")


def main():
    config = Config.from_yaml("test/train_llama3.yaml")

    if config.train is None:
        raise RuntimeError("Train config is not set")

    train = UnslothTrainer(config.train.experiments[0], get_dataset_pth())
    train.train()


if __name__ == "__main__":
    main()
