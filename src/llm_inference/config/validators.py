# This module contains the validators for the config files.
# At this moment the only validator is validModelUrl and it test that the url is not empty.


class Validator:
    @staticmethod
    def valid_model_url(url: str) -> str:
        if not url:
            raise ValueError(f"Validator:validModelUrl| Invalid model url: {url}")

        return url
