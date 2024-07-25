from typing import override

import torch

# From hf
from datasets import Dataset
from llm_inference.config.config import Config
from llm_inference.dataset.hf_msg_dataset import HfMsgDataset
from llm_inference.runner.abstract_model_runner import AbstractModelRunner
from llm_inference.runner.model_output_item import ModelOutputItem
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.tokenization_utils_base import BatchEncoding


class HFRunner(AbstractModelRunner):
    def __init__(self, config: Config):
        super().__init__(config)
        self.llm_model = self._get_model()
        self.generation_config = self._get_generation_config()

    def _get_generation_config(self) -> GenerationConfig:
        model_conf = self.config.llm_model
        return GenerationConfig(
            max_new_tokens=model_conf.max_new_tokens,
            eos_token_id=model_conf.terminators,
            do_sample=model_conf.do_sample,
            temperature=model_conf.temperature,
            top_p=model_conf.top_p,
            pad_token_id=model_conf.pad_token_id,
        )

    def _get_model(self):  # -> Callable[..., Any]:
        llm_model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model.llm_url,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # Attention works only with bfloat16 or float16
            # Need move to config
            # attn_implementation="flash_attention_2",
            attn_implementation="sdpa",
        )
        # For this you need to install optimum.
        # WARNING: This is not supported and not tested yet.
        # llm_model = llm_model.to_bettertransformer()
        torch.set_float32_matmul_precision("high")
        return torch.compile(llm_model).eval()

    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.llm_model.llm_url)

    def _generate_tokens(self, model_input_tokens: BatchEncoding) -> torch.Tensor:
        return self.llm_model.generate(
            **model_input_tokens,
            generation_config=self.generation_config,
        )

    @override
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

    # WARNING: NOT TESTED
    # This need to refactor. a lot of problems here.
    @override
    @torch.no_grad()
    def execute(self, dataset: HfMsgDataset) -> list[ModelOutputItem]:
        # model_input.tokenize(lambda x: self.tokenizer(x, return_tensors="pt"))
        dataloader: DataLoader[Dataset] = DataLoader(
            dataset.get_hf_dataset(),
            shuffle=False,
            batch_size=self.model_config.dataset.batch_size,
            num_workers=self.config.general.num_workers,
            # TODO: Need check why not work with pin_memory
            # pin_memory=False,
            # pin_memory_device="cuda",
        )

        model_output: list[ModelOutputItem] = []
        # For this need flash attention, not needed with torch >= 2.1.1
        # with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
        for batch_num, batch_tokens in enumerate(tqdm(dataloader)):
            # BUG: Need to fix this, we don't need to use `to`, because memory management is handled by dataloader.
            cnt_tokens: int = batch_tokens["input_ids"][0].shape[0]
            output_tokens = self._generate_tokens(batch_tokens).to("cpu")
            output_strings: list[str] = dataset.batch_decode(output_tokens, cnt_tokens)

            first_index_ds = batch_num * self.model_config.dataset.batch_size
            last_index_ds = first_index_ds + self.model_config.dataset.batch_size
            model_groups = [x.group_id for x in dataset[first_index_ds:last_index_ds]]
            model_output.extend(
                [
                    ModelOutputItem(group_id=group_id, text=res)
                    for res, group_id in zip(output_strings, model_groups)
                ]
            )
            torch.save(
                output_tokens,
                self.config.general.backup_path
                / f"output_tokens_backup(batch - {batch_num}).pt",
            )

        return model_output
