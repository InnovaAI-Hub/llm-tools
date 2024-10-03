from abc import abstractmethod
from pydantic import BaseModel


class AbstractTrainer(BaseModel):
    @abstractmethod
    def train(self):
        raise NotImplementedError
