"""
Description:
    In this file we define the type of the model.
Classes:
    - ModelType: A class for storing the type of the model.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 03.03.2025
Version: 0.3
Python Version: 3.11.9
"""

from enum import StrEnum


class ModelType(StrEnum):
    LLAMA3 = "llama-3"
    LLAMA3_VISION = "llama-3-vision"
    GEMMA3 = "gemma3"
    # TODO: Add other model types

    @staticmethod
    def get_models_dict() -> dict["ModelType", dict[str, bool]]:
        """
        Returns a dictionary mapping model types to their properties.

        Returns:
            dict[ModelType, dict[str, bool]]: A dictionary where keys are ModelType instances
            and values are dictionaries containing model properties.
        """
        data = {
            ModelType.LLAMA3: {"is_vision": False},
            ModelType.LLAMA3_VISION: {"is_vision": True},
            ModelType.GEMMA3: {"is_vision": True},
        }

        return data

    @staticmethod
    def get_vision_models_list() -> list["ModelType"]:
        """
        Returns a list of model types that have vision capabilities.

        Returns:
            list[ModelType]: A list of ModelType instances that are vision models.
        """
        all_models = ModelType.get_models_dict()
        vision_models = [
            model for model in all_models if all_models[model]["is_vision"]
        ]
        return vision_models

    @staticmethod
    def it_vision_models(model_type: "ModelType") -> bool:
        """
        Checks if a given model type has vision capabilities.

        Args:
            model_type (ModelType): The model type to check.

        Returns:
            bool: True if the model type is a vision model, False otherwise.
        """

        return model_type in ModelType.get_vision_models_list()

    @staticmethod
    def it_chat_models(model_type: "ModelType") -> bool:
        """
        Checks if a given model type has chat capabilities.

        Args:
            model_type (ModelType): The model type to check.

        Returns:
            bool: True if the model type is a chat model, False otherwise.
        """
        return not ModelType.it_vision_models(model_type)
