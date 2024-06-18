from llm_inference.config.config import Config
from llm_inference.runner.abstract_model_runner import AbstractModelRunner
from vllm import LLM, SamplingParams


# WARNING: This runner is not fully supported yet, not tested and should not be used.
class VLLMRunner(AbstractModelRunner):
    def __init__(self, model_config: Config) -> None:
        super().__init__(model_config)
        self.params = SamplingParams(
            temperature=self.model_config.temperature,
            top_p=self.model_config.top_p,
            max_tokens=self.model_config.max_new_tokens,
            stop_token_ids=self.model_config.terminators,
        )
        self.model = LLM(self.model_config.llm_url, dtype="bfloat16")

    def execute(self, input: str) -> str:
        model_output = self.model.generate(
            input, sampling_params=self.params, use_tqdm=True
        )

        if not len(model_output):
            raise RuntimeWarning("VLLMRunner::execute| Model output is empty")

        return model_output[0].outputs[0].text
