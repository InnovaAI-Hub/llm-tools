from enum import StrEnum


class MetricType(StrEnum):
    """Types of metrics"""

    HF_METRIC = "hf_metric"
    SVG_SSIM_METRIC = "svg_ssim_metric"
