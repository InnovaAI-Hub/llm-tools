"""
Description:
    In this file we define the runner class for Huggingface models.
    Can be used also for model with peft adapters.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 03.10.2024
Version: 0.3
Python Version: 3.12
Dependencies:
    - pydantic
    - transformers
    - torch
    - tqdm
    - peft
    - datasets
"""

# from typing import override
from typing_extensions import deprecated

import torch

from datasets import Dataset  # By Huggingface
from llm_tools.config.config import Config
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from llm_tools.llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_tools.llm_inference.runner.model_output_item import ModelOutputItem
from peft import PeftModel  # By Huggingface
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
)  # By Huggingface
from transformers.tokenization_utils_base import BatchEncoding  # By Huggingface


class HFRunner(AbstractModelRunner):
    def __init__(self, config: Config):
        super().__init__(config)
        self.llm_model = self._get_model()
        self.generation_config = self._get_generation_config()
        # TODO: Where will be better to init tokenizer?

    def __del__(self):
        torch.cuda.empty_cache()

    def _get_generation_config(self) -> GenerationConfig:
        model_conf = self.config.llm_model
        return GenerationConfig(
            max_new_tokens=model_conf.max_new_tokens,
            eos_token_id=model_conf.terminators,
            do_sample=model_conf.do_sample,
            temperature=model_conf.temperature,
            top_p=model_conf.top_p,
            pad_token_id=model_conf.pad_token_id,  # TODO: It's problem. It's not the same as tokenizer.pad_token_id
        )

    def _get_model(self):  # -> Callable[..., Any]:
        torch.set_float32_matmul_precision("high")

        llm_model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model.llm_url,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # Attention works only with bfloat16 or float16
            # Need move to config
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )

        if self.config.llm_model.peft_path is not None:
            if self.config.llm_model.resize_embed_layer is not None:
                llm_model.resize_token_embeddings(
                    self.config.llm_model.resize_embed_layer
                )

            self.logger.info("HFRunner::_get_model| Loading peft adapter.")
            llm_model: PeftModel = PeftModel.from_pretrained(
                llm_model,
                self.config.llm_model.peft_path,
                is_trainable=False,
                device="cuda",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self.logger.info("HFRunner::_get_model| Adapter loaded.")

            return llm_model.merge_and_unload(progressbar=True, safe_merge=True)

        # For this you need to install `optimum`.
        # WARNING: This is not supported and not tested yet.
        # llm_model = llm_model.to_bettertransformer()
        return torch.compile(llm_model).eval()

    def _get_tokenizer(self):
        # TODO: We actually have variable with tokenaizer_pth. We need also to check this.
        return AutoTokenizer.from_pretrained(self.config.llm_model.llm_url)

    def _generate_tokens(self, model_input_tokens: BatchEncoding) -> torch.Tensor:
        return self.llm_model.generate(
            **model_input_tokens,
            generation_config=self.generation_config,
        )

    @deprecated(
        "HFRunner::execute_once| This method in refactor, use `execute` instead."
    )
    # @override
    @torch.no_grad()
    def execute_once(self, model_input: str) -> str:
        tokenizer = self._get_tokenizer()

        # TODO: Add support for different device
        tokens: BatchEncoding = tokenizer(model_input, return_tensors="pt").to("cuda")

        self.logger.debug("HFRunner::execute| Start model execution.")
        model_output_tokens = self._generate_tokens(tokens)
        self.logger.debug("HFRunner::execute| End model execution.")

        if not len(model_output_tokens):  # type: ignore
            raise RuntimeWarning("HFRunner::execute| Model output is empty")

        # TODO: Try change to .decode(). UPD: Need to test.
        model_output = tokenizer.decode(model_output_tokens)

        return model_output

    # @override
    @torch.no_grad()
    def execute(self, dataset: HfMsgDataset) -> list[ModelOutputItem]:
        # model_input.tokenize(lambda x: self.tokenizer(x, return_tensors="pt"))
        """
        Execute a model on a given dataset.

        Args:
            dataset (HfMsgDataset): The dataset to run the model on.

        Returns:
            list[ModelOutputItem]: A list of ModelOutputItem objects, each containing the output text and the group id of the input.
        """
        ds = dataset.get_hf_dataset()

        all_cols = set(ds.column_names)
        # TODO: Get this set from model config

        group_ids = ds["group_id"]

        used_cols = {"input_ids", "attention_mask"}
        ds = ds.remove_columns(list(all_cols - used_cols))

        dataloader: DataLoader[Dataset] = DataLoader(
            dataset=ds,
            shuffle=False,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.environment.num_workers,
            # TODO: Need check why not work with pin_memory
            pin_memory=True,
            pin_memory_device="cuda",
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=dataset.tokenizer, mlm=False
            ),
        )

        model_output_str: list[str] = []
        # For this need flash attention, not needed with torch >= 2.1.1
        # with torch.nn.attention.sdpa_kernel(
        #     backends=torch.nn.attention.SDPBackend.FLASH_ATTENTION
        # ):
        for batch_num, batch_tokens in enumerate(tqdm(dataloader, desc="Run")):
            batch_tokens: BatchEncoding
            # model_groups: list[int | str] = batch_tokens.data.pop("group_id")

            batch_tokens = batch_tokens.to("cuda")
            cnt_tokens: int = batch_tokens["input_ids"][0].shape[0]

            self.logger.debug(
                f"HFRunner::execute| Start model execution. Batch number: {batch_num}, cnt_tokens: {cnt_tokens}",
            )

            output_tokens = self._generate_tokens(batch_tokens).to("cpu")
            output_strings: list[str] = dataset.batch_decode(output_tokens, cnt_tokens)

            model_output_str.extend(output_strings)

            torch.save(
                output_tokens,
                self.config.environment.backup_path
                / f"output_tokens_backup(batch - {batch_num}).pt",
            )

        model_output = [
            ModelOutputItem(group_id=group_id, text=res)
            for res, group_id in zip(model_output_str, group_ids)
        ]

        return model_output
