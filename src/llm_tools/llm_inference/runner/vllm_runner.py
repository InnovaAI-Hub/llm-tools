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
Version: 0.2
Python Version: 3.12.9
Dependencies:
    - pydantic
    - vllm
    - torch

TODO:
    - After a lot updates of hf dataset, need to check the work of all methods.
"""

# from typing import override

import numpy as np
import pandas as pd
import ray
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.outputs import RequestOutput

from llm_tools.auto_tokenizer_processor.abstract_wrapper import AbstractTokenizerWrapper
from llm_tools.auto_tokenizer_processor.selector import select_tokenizer_processor
from llm_tools.config.config import Config
from llm_tools.dataset.dataset import Dataset
from llm_tools.llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_tools.llm_inference.runner.model_output_item import ModelOutputItem


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

        self.tokenizer = self._get_tokenizer()
        self.model.set_tokenizer(self.tokenizer.get_current_tokenizer())

    def __del__(self) -> None:
        destroy_model_parallel()
        destroy_distributed_environment()

        torch.cuda.empty_cache()
        ray.shutdown()

    #    @override
    def execute_once(self, model_input: str) -> str:
        model_output = self.model.generate(
            model_input, sampling_params=self.params, use_tqdm=True
        )

        if not len(model_output):
            raise RuntimeWarning("VLLMRunner::execute| Model output is empty")

        return model_output[0].outputs[0].text

    def _get_tokenizer(self) -> AbstractTokenizerWrapper:
        return select_tokenizer_processor(self.config)

    #    @override
    def execute(self, model_input_ds: Dataset) -> list[ModelOutputItem]:
        batch_size: int = self.config.dataset.batch_size
        count_batches: int = max(1, len(model_input_ds) // batch_size)
        generation_result: list[ModelOutputItem] = []

        model_inputs: list[dict[str, str]] = [
            {"sentence": item.format_dialog(self.tokenizer), "group_id": item.group_id}
            for item in tqdm(model_input_ds)
        ]

        for batch_num, batch in enumerate(
            tqdm(
                np.array_split(model_inputs, count_batches),
                desc="Processed batch",
                total=count_batches,
            )  # type: ignore
        ):
            sentences: list[str] = [x["sentence"] for x in batch]
            groups: list[int | str] = [x["group_id"] for x in batch]

            model_output_tmp: list[RequestOutput] = self.model.generate(
                prompts=sentences,
                sampling_params=self.params,
                use_tqdm=True,
            )

            batch_generation_result: list[ModelOutputItem] = [
                ModelOutputItem(group_id, item.outputs[0].text)
                for item, group_id in zip(model_output_tmp, groups)
            ]

            if self.config.environment.backup_path is not None:
                backup_dir = self.config.environment.backup_path
                tmp_backup_res = pd.DataFrame(
                    {
                        "groups": groups,
                        "content": [item.text for item in batch_generation_result],
                    }
                )
                tmp_backup_res.to_parquet(
                    backup_dir / f"output_backup(batch - {batch_num}).parquet"
                )

            generation_result.extend(batch_generation_result)

        return generation_result
