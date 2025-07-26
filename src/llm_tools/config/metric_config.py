from pydantic import BaseModel, Field
from llm_tools.type.metric import MetricType


class MetricConfig(BaseModel):
    metric_type: MetricType
    name: str = Field(default="Unnamed Metric")
