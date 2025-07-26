from llm_tools.config.metric_config import MetricConfig
from llm_tools.type.metric import MetricType

from .hf_metric_wrapper import HFMetricWrapper
from .svg_img_metric import SVGSSIMMetric
from .abstract_metric import AbstractMetric


def select_metric(metric_config: MetricConfig) -> AbstractMetric:
    metrics = {
        MetricType.HF_METRIC: HFMetricWrapper,
        MetricType.SVG_SSIM_METRIC: SVGSSIMMetric,
    }

    return metrics[metric_config.metric_type](metric_config.name)
