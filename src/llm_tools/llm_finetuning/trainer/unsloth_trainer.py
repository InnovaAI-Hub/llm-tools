"""
Module for training Hugging Face models using the UnslothTrainer class.

Description:
    This module provides a class, UnslothTrainer, for fine-tuning Hugging Face models.
    It handles loading the dataset, creating the model, and training the model.

Classes:
    UnslothTrainer: A class for fine-tuning Hugging Face models.

Author: Artem Durynin
E-mail: artem.durynin@raftds.com, mail@durynin1.ru
Date Created: 10.09.24
Date Modified: 03.12.24
Version: 0.1
Python Version: 3.10
Dependencies:
    - Hugging Face: Transformers, evaluate, peft, trl;
    - PyTorch
    - pandas
    - numpy
    - unsloth
"""

import logging
import os

# from typing import override # Only in python 12.

import evaluate
import numpy as np
import torch
from llm_tools.config.experiment_config import ExperimentConfig
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth_zoo import tokenizer_utils
from pathlib import Path

from llm_tools.llm_finetuning.trainer.abstract_trainer import (
    AbstractTrainer,
    PreparedDataset,
)


class UnslothTrainer(AbstractTrainer):
    # NOTE: In this moment we use default dataset and after it we convert it needed format.
    #       Maybe need to change this?
    def __init__(self, exp_config: ExperimentConfig, ds_path: Path) -> None:
        super().__init__()

        self.exp_config: ExperimentConfig = exp_config
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.model, self.tokenizer = self._get_model()
        self.dataset: PreparedDataset = self.get_dataset(ds_path, self.tokenizer)
        self.evaluator: evaluate.EvaluationModule = evaluate.load(
            self.exp_config.metric
        )

    # @override
    def evaluate(self, predict_labels: EvalPrediction):
        """
        Compute the evaluation metric for the given `predict_labels`.

        Args:
            predict_labels: A tuple of (logits, labels) where logits is the prediction
                output from the model and labels is the original labels.

        Returns:
            A dictionary containing the evaluation metric and the metric value.
        """
        # TODO: Need to add check that pad_token_id is not equal `none`.
        logits = predict_labels.predictions
        labels = predict_labels.label_ids
        self.logger.debug(
            f"UnslothTrainer::evaluation| Predict labels type: {type(predict_labels)}\n"
            f"UnslothTrainer::evaluation| Got logits and labels, types: {type(logits)}, {type(labels)}."
        )

        # TODO: Remove or improve this check.
        if type(logits) is tuple:
            self.logger.debug(
                f"UnslothTrainer::evaluation| Logits is tuple:\n* Size:{len(logits)},\n* Items:{logits}"
            )
            assert (
                len(logits) != 0
            ), "UnslothTrainer::evaluation| Logits is empty tuple."

        if type(labels) is tuple:
            self.logger.debug(
                f"UnslothTrainer::evaluation| Labels is tuple:\n* Size:{len(labels)},\n* Items:{labels}"
            )
            assert (
                len(labels) != 0
            ), "UnslothTrainer::evaluation| Labels is empty tuple."

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

    def _get_model(self) -> tuple[FastLanguageModel, PreTrainedTokenizerFast]:
        # torch.set_float32_matmul_precision("high")
        """
        This method returns a tuple with the model and the tokenizer.
        The model is a FastLanguageModel from the unsloth library.
        The tokenizer is a PreTrainedTokenizerFast from the Hugging Face library.

        Returns:
            tuple[FastLanguageModel, PreTrainedTokenizerFast]: The method returns a tuple with the model and the tokenizer.
        """
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        model_conf = self.exp_config.llm_model
        train_conf = self.exp_config.training_arguments
        lora_conf = self.exp_config.peft_method.lora_conf

        model_tokenizer = FastLanguageModel.from_pretrained(
            self.exp_config.llm_model.llm_url,
            # TODO: add config support
            dtype=torch.float16 if train_conf.fp16 else torch.bfloat16,  # Not good...
            device_map="auto",
            max_seq_length=model_conf.max_seq_length,
            load_in_4bit=False,
        )

        model: FastLanguageModel = model_tokenizer[0]
        tokenizer: PreTrainedTokenizerFast = model_tokenizer[1]

        if self.exp_config.additional_tokens is not None:
            new_tokens = [
                token.token_name for token in self.exp_config.additional_tokens
            ]

            tokenizer_utils.add_new_tokens(
                model,
                tokenizer,
                new_tokens=new_tokens,
                method="mean",
                interpolation=0.5,
            )

            self.logger.info(
                "UnslothTrainer::_get_model| Added new tokens to the tokenizer, current length: %d",
                len(tokenizer),
            )

        self.logger.debug("Model:%s", model)

        return FastLanguageModel.get_peft_model(
            model,
            r=lora_conf.r,
            lora_alpha=lora_conf.lora_alpha,
            lora_dropout=lora_conf.lora_dropout,
            modules_to_save=lora_conf.modules_to_save,
            target_modules=lora_conf.target_modules,
            init_lora_weights=True if lora_conf.init_lora_weights else False,
            bias=lora_conf.bias,
            layers_pattern=lora_conf.layers_pattern,
            use_rslora=lora_conf.use_rslora,
            loftq_config=lora_conf.loftq_config,
            use_gradient_checkpointing=train_conf.gradient_checkpointing,
            layers_to_transform=lora_conf.layers_to_transform,
            random_state=13,
            # temporary_location=
        ), tokenizer

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
            packing=False,
            dataset_text_field="input_sentences",
            train_dataset=self.dataset.train.shuffle(13),
            eval_dataset=self.dataset.test.shuffle(13),
            compute_metrics=self.evaluate,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ),
        )

        trainer.train()

        output_path = self.exp_config.save_to / self.exp_config.experiment_name
        output_path.mkdir(exist_ok=True, parents=True)
        trainer.save_model(output_path.as_posix())

        if self.exp_config.save_merged:
            merged_dir = output_path / "merged_model"
            merged_dir.mkdir(exist_ok=True, parents=True)
            merged_model = trainer.model.merge_and_unload(
                progressbar=True, safe_merge=True
            )
            merged_model.save_pretrained(merged_dir.as_posix())
            self.tokenizer.save_pretrained(merged_dir.as_posix())
