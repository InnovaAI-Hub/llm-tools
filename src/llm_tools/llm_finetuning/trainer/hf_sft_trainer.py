import logging
from pathlib import Path
# from typing import override

import evaluate
import numpy as np
import torch
from llm_tools.config.experiment_config import ExperimentConfig
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from llm_tools.llm_finetuning.trainer.abstract_trainer import (
    AbstractTrainer,
    PreparedDataset,
)
from peft.mapping import get_peft_model
from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer


class HFSFTTrainer(AbstractTrainer):
    # NOTE: In this moment we use default dataset and after it we convert it needed format.
    #       Maybe need to change this?
    def __init__(self, exp_config: ExperimentConfig, ds_path: Path) -> None:
        super().__init__()

        self.exp_config: ExperimentConfig = exp_config
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.model = self._get_model()
        self.tokenizer = HfMsgDataset.get_tokenizer(self.exp_config.llm_model.llm_url)
        self.dataset: PreparedDataset = self.get_dataset(ds_path, self.tokenizer)
        self.evaluator: evaluate.EvaluationModule = evaluate.load(
            self.exp_config.metric
        )

    #    @override
    def evaluate(self, predict_labels):
        """
        Compute the evaluation metric for the given `predict_labels`.

        Args:
            predict_labels: A tuple of (logits, labels) where logits is the prediction
                output from the model and labels is the original labels.

        Returns:
            A dictionary containing the evaluation metric and the metric value.
        """
        # TODO: Need to add check that pad_token_id is not equal `none`.
        logits, labels = predict_labels

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "UnslothTrainer::evaluation| The tokenizer does not have a pad token id."
            )

        skip_token = -100
        labels = np.where(labels != skip_token, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        predict = np.argmax(logits, axis=-1)
        decoded_predictions = self.tokenizer.batch_decode(
            predict, skip_special_tokens=True
        )

        return self.evaluator.compute(
            predictions=decoded_predictions, references=decoded_labels
        )

    def _get_model(self) -> PeftModel | PeftMixedModel:
        model_conf = self.exp_config.llm_model
        train_conf = self.exp_config.training_arguments
        peft_conf = self.exp_config.peft_method.lora_conf

        model = AutoModelForCausalLM.from_pretrained(
            model_conf.llm_url,
            # TODO: add config support
            torch_dtype=torch.float16
            if train_conf.fp16
            else torch.bfloat16,  # Not good...
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.logger.debug("Model:%s", model)

        return get_peft_model(model, peft_conf)

    def train(self) -> None:
        """
        Start training model.
        Model will be saved in `exp_config.save_to / exp_config.experiment_name`.
        """

        self.logger.info("train:: Start training")

        sft_conf = SFTConfig(**self.exp_config.training_arguments.to_dict())

        trainer = SFTTrainer(
            model=self.model,
            args=sft_conf,
            tokenizer=self.tokenizer,
            packing=False,
            dataset_text_field="input_sentences",
            train_dataset=self.dataset.train.shuffle(13),
            eval_dataset=self.dataset.test.shuffle(13),
            compute_metrics=self.evaluate,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
            max_seq_length=self.exp_config.llm_model.max_seq_length,
        )

        trainer.train()

        output_path = self.exp_config.save_to / self.exp_config.experiment_name
        output_path.mkdir(exist_ok=True, parents=True)
        trainer.save_model(output_path.as_posix())
