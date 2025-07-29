"""
Description:
    In this file we define the type of the url.

Goal:
    It's provide the type of the url with validation.
    Validation:
        * If is path, that check that the path exists.
        * If is file, that check that the file exists.
        * If is a url, that check that the url is valid.
        * If is hf url, that check that the hf url is valid and model is available.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 13.10.2024
Version: 0.1
Python Version: 3.10

"""

from llm_tools.config.validators import Validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated


UrlType = Annotated[str, AfterValidator(Validator.valid_model_url)]
