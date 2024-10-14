"""
Description:
    In this file we define the abstract trainer class.
    Need a lot refactoring.

Author: Artem Durynin
E-mail: artem.d@raftds.com, mail@durynin1.ru
Date Created: 19.09.2024
Date Modified: 19.06.2024
Version: 0.1
Python Version: 3.12
Dependencies:
    - pydantic
"""

from abc import abstractmethod
from pydantic import BaseModel


class AbstractTrainer(BaseModel):
    @abstractmethod
    def train(self):
        raise NotImplementedError
