import torch
from llm_inference.config.config import Config
from llm_inference.dataset.msg_dataset import MsgDataset
from llm_inference.runner.abstract_model_runner import AbstractModelRunner
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
        self.tokenizer = self._get_tokenizer()
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
            self.config.llm_model.llm_url, device_map="auto", torch_dtype=torch.bfloat16
        )
        torch.set_float32_matmul_precision("high")
        return torch.compile(llm_model).eval()

    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.config.llm_model.llm_url)

    def _generate_tokens(self, model_input_tokens: BatchEncoding) -> torch.Tensor:
        return self.llm_model.generate(
            **model_input_tokens, generation_config=self.generation_config
        )

    @torch.no_grad()
    def execute(self, model_input: str) -> str:
        # TODO: Add support for different device
        tokens: BatchEncoding = self.tokenizer(model_input, return_tensors="pt").to(
            "cuda"
        )

        self.logger.debug("HFRunner::execute| Start model execution.")
        model_output_tokens = self._generate_tokens(tokens)
        self.logger.debug("HFRunner::execute| End model execution.")

        if not len(model_output_tokens):  # type: ignore
            raise RuntimeWarning("HFRunner::execute| Model output is empty")

        # TODO: Try change to .decode()
        model_output = self.tokenizer.batch_decode(model_output_tokens)

        return model_output[0]

    # WARNING: NOT TASTED
    @torch.no_grad()
    def execute_batch(self, model_input: MsgDataset) -> list[str]:
        model_input_tokens = model_input.tokenize(self.tokenizer)
        model_output_tokens = [
            self._generate_tokens(tokens) for tokens in tqdm(model_input_tokens)
        ]
        results = model_input.batch_decode(self.tokenizer, model_output_tokens)
        return results
