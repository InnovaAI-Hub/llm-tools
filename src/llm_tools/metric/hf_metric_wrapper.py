from typing import override
from .abstract_metric import AbstractMetric
import evaluate


class HFMetricWrapper(AbstractMetric):
    def __init__(self, metric_name: str | None = None):
        super().__init__(metric_name)

        if not self.metric_name:
            raise ValueError("HF Metric name not provided.")

        self.metric = evaluate.load(self.metric_name)

    @override
    def compute(self, reference, predictions):
        return self.metric.compute(predictions=predictions, references=reference)
