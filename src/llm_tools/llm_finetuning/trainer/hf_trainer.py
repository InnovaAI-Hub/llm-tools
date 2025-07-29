"""
Module for training Hugging Face models using the HFTrainer class.

Description:
    This module provides a class, HFTrainer, for fine-tuning Hugging Face models.
    It handles loading the dataset, creating the model, and training the model.

Classes:
    HFTrainer: A class for fine-tuning Hugging Face models.

Author: Artem Durynin
E-mail: artem.durynin@raftds.com, mail@durynin1.ru
Date Created: 10.09.24
Date Modified: 11.10.24
Version: 0.1
Python Version: 3.10
Dependencies:
    - Hugging Face: Transformers, evaluate, peft;
    - PyTorch
    - pandas
    - numpy
"""

import logging
from pathlib import Path
# from typing import override

import evaluate
import numpy as np
import torch
from llm_tools.config.experiment_config import ExperimentConfig
from llm_tools.dataset.hf_msg_dataset import HfMsgDataset
from llm_tools.llm_finetuning.trainer.abstract_trainer import AbstractTrainer
from peft.config import PeftConfig
from peft.mapping import get_peft_model
from peft.mixed_model import PeftMixedModel
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
)


# WARNING: Not use this class. Need to refactor this.
class HFTrainer(AbstractTrainer):
    # NOTE: In this moment we use default dataset and after it we convert it needed format.
    #       Maybe need to change this?
    def __init__(self, exp_config: ExperimentConfig, ds_path: Path) -> None:
        super().__init__()

        self.exp_config = exp_config
        self.logger = logging.getLogger(__name__)
        self.model = self._get_model(exp_config.peft_method.lora_conf)
        self.tokenizer = HfMsgDataset.get_tokenizer(self.exp_config.llm_model.llm_url)
        self.dataset = self.get_dataset(ds_path, self.tokenizer)
        self.evaluator = evaluate.load(self.exp_config.metric)

    #    @override
    def evaluate(self, predict_labels):
        logits, labels = predict_labels
        # TODO: Need to add check that pad_token_id is not equal `none`.

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "HFTrainer::evaluation| The tokenizer does not have a pad token id."
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

    # TODO: Rewrite this to static?
    def _get_model(self, peft_config: PeftConfig) -> PeftModel | PeftMixedModel:
        model_conf = self.exp_config.llm_model
        train_conf = self.exp_config.training_arguments

        model = AutoModelForCausalLM.from_pretrained(
            model_conf.llm_url,
            # TODO: add config support
            torch_dtype=torch.float16 if train_conf.fp16 else torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.logger.debug("Model:%s", model)

        return get_peft_model(model, peft_config)

    def log_model_parameters(self):
        trainable_params, all_param = self.model.get_nb_trainable_parameters()
        self.logger.info(
            f"log_model_parameters:: Trainable params: {trainable_params} | All parameters: {all_param}"
        )

    def train(self):
        self.logger.info("train:: Start training")
        self.log_model_parameters()

        trainer = Trainer(
            self.model,
            self.exp_config.training_arguments,
            train_dataset=self.dataset["train"].shuffle(13),
            eval_dataset=self.dataset["test"].shuffle(13),
            compute_metrics=self.evaluate,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
        )

        trainer.train()

        output_path = self.exp_config.save_to / self.exp_config.experiment_name
        output_path.mkdir(exist_ok=True, parents=True)
        trainer.save_model(output_path.as_posix())
