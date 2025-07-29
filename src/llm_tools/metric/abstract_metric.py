from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    def __init__(self, metric_name: str | None = None):
        self.metric_name: str | None = metric_name

    @abstractmethod
    def compute(self, reference, predictions):
        raise NotImplementedError
