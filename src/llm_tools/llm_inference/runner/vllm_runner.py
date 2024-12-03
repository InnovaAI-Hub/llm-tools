"""
Description:
    This file defines the VLLMRunner class, which is used for running models with VLLM.
    Currently, PEFT adapters are not supported. The class can be used with Hugging Face models.

Classes:
    - VLLMRunner: A class for running VLLM models with configurable sampling parameters and input datasets.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 3.12.2024
Version: 0.1
Python Version: 3.10
Dependencies:
    - pydantic
    - vllm
    - torch

TODO:
    - After a lot updates of hf dataset, need to check the work of all methods.
"""

# from typing import override

import torch

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

        self.model = LLM(
            self.model_config.llm_url,
            dtype="bfloat16",
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=self.model_config.max_model_len,
        )

    #    @override
    def execute_once(self, input: str) -> str:
        model_output = self.model.generate(
            input, sampling_params=self.params, use_tqdm=True
        )

        if not len(model_output):
            raise RuntimeWarning("VLLMRunner::execute| Model output is empty")

        return model_output[0].outputs[0].text

    #    @override
    def execute(self, input_ds: HfMsgDataset) -> list[ModelOutputItem]:
        # Try with one batch.
        # batch_list = [item for item in input_ds]
        self.model.set_tokenizer(
            input_ds.tokenizer
        )  # TODO: Why we set tokenizer here? Maybe move it to __init__.

        model_output_tmp = self.model.generate(
            prompts=[x.sentence for x in input_ds],
            sampling_params=self.params,
            use_tqdm=True,
        )

        model_output: list[ModelOutputItem] = [
            ModelOutputItem(group_id, item.outputs[0].text)
            for item, group_id in zip(model_output_tmp, [x.group_id for x in input_ds])
        ]

        return model_output
