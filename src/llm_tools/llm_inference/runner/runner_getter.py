"""
Description:
    In this file we define the runner getter class.
    It is used to get the runner based on the runner type.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 13.10.2024
"""

from llm_tools.config.config import Config
from llm_tools.llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_tools.llm_inference.runner.server_runner import ServerRunner
from llm_tools.type.runner_type import RunnerType
from llm_tools.llm_inference.runner.hf_runner import HFRunner
from llm_tools.llm_inference.runner.vllm_runner import VLLMRunner
from llm_tools.llm_inference.runner.unsloth_runner import UnslothRunner


class RunnerGetter:
    @staticmethod
    def get_runner(runner_type: RunnerType, config: Config) -> AbstractModelRunner:
        runners = {
            RunnerType.HF: HFRunner,
            RunnerType.VLLM: VLLMRunner,
            RunnerType.UNSLOTH: UnslothRunner,
            RunnerType.SERVER: ServerRunner,
        }

        return runners[runner_type](config)
