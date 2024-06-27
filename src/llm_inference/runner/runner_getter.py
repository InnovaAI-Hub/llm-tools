from llm_inference.config.config import Config
from llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_inference.type.runner_type import RunnerType
from llm_inference.runner.hf_runner import HFRunner
from llm_inference.runner.vllm_runner import VLLMRunner


class RunnerGetter:
    @staticmethod
    def get_runner(runner_type: RunnerType, config: Config) -> AbstractModelRunner:
        runners = {
            RunnerType.HF: HFRunner,
            RunnerType.VLLM: VLLMRunner,
        }

        return runners[runner_type](config)
