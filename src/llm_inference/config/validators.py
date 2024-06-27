"""
Description: This module contains the validators for the config files.
    At this moment the only validator is validModelUrl and it test that the url is not empty.
Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 13.06.2024
Date Modified: 14.06.2024
Version: 0.1
Python Version: 3.10
Dependencies: None
"""


class Validator:
    @staticmethod
    def valid_model_url(url: str) -> str:
        if not url:
            raise ValueError(f"Validator:validModelUrl| Invalid model url: {url}")

        return url
