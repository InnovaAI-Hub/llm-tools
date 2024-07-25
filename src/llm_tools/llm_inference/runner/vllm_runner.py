from typing import override

from llm_tools.config.config import Config
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from llm_tools.llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_tools.llm_inference.runner.model_output_item import ModelOutputItem
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

    @override
    def execute_once(self, input: str) -> str:
        model_output = self.model.generate(
            input, sampling_params=self.params, use_tqdm=True
        )

        if not len(model_output):
            raise RuntimeWarning("VLLMRunner::execute| Model output is empty")

        return model_output[0].outputs[0].text

    @override
    def execute(self, input: HfMsgDataset) -> list[ModelOutputItem]:
        # Use or not batches and dataloader?

        # Try with one batch.
        batch_list = [item for item in input]

        model_output_tmp = self.model.generate(
            prompts=[x.sentence for x in batch_list],
            sampling_params=self.params,
            use_tqdm=True,
        )

        model_output: list[ModelOutputItem] = [
            ModelOutputItem(group_id, item.outputs[0].text)
            for item, group_id in zip(
                model_output_tmp, [x.group_id for x in batch_list]
            )
        ]

        return model_output
