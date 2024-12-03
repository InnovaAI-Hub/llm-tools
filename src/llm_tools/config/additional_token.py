from pydantic import Field
from pydantic_settings import BaseSettings
import tokenizers


class AdditionalToken(BaseSettings):
    # TODO: Switch to hf class.
    """
    Additional token settings.
    Please, check: https://huggingface.co/docs/tokenizers/api/added-tokens
    """

    token_name: str = Field(frozen=True)
    single_word: bool = Field(default=False, frozen=True)
    lstrip: bool = Field(default=False, frozen=True)
    rstrip: bool = Field(default=False, frozen=True)
    normalized: bool = Field(default=True, frozen=True)
    special: bool = Field(default=False, frozen=True)

    def conv_to_hf(self) -> tokenizers.AddedToken:
        return tokenizers.AddedToken(
            content=self.token_name,
            single_word=self.single_word,
            lstrip=self.lstrip,
            rstrip=self.rstrip,
            normalized=self.normalized,
            special=self.special,
        )
