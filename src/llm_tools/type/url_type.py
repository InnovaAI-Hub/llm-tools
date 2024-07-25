from llm_tools.config.validators import Validator
from pydantic.functional_validators import AfterValidator
from typing import Annotated


UrlType = Annotated[str, AfterValidator(Validator.valid_model_url)]
