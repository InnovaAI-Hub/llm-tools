import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from llm_inference.config.config import Config
from llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_inference.runner.runner_getter import RunnerGetter


@dataclass(slots=True)
class MainApplication:
    logger = logging.getLogger(__name__)

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser("LLM Inference")
        parser.add_argument(
            "--config", type=str, required=True, help="Path to config file"
        )
        return parser.parse_args()

    def run(self) -> None:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self.logger.info("MainApplication::run| Start application.")
        try:
            args: argparse.Namespace = self.parse_args()
            conf = Config.from_yaml(args.config)
            text = Path("./src/llm_inference/test_prompt.dt").read_text()
            runner: AbstractModelRunner = RunnerGetter.get_runner(
                conf.general.runner_type
            )(conf)
            print(runner.execute(text))
        except Exception as error:
            self.logger.critical("MainApplication::run: %s", error, exc_info=True)
