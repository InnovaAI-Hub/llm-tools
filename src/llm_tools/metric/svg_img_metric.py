from io import BytesIO
from typing import override

import logging
import numpy as np
from cairosvg import svg2png
from skimage import io
from skimage.metrics import structural_similarity as ssim

from .abstract_metric import AbstractMetric

logger = logging.getLogger(__name__)


class SVGSSIMMetric(AbstractMetric):
    def __init__(self, metric_name: str | None = "SSIM"):
        super().__init__(metric_name)

    @staticmethod
    def get_svg(full_text: str):
        svg_index_start = full_text.rfind("<svg")
        svg_index_end = full_text.rfind("</svg>")
        svg = full_text[svg_index_start : svg_index_end + len("</svg>")]
        return svg

    def _compute(self, reference: str, predictions: str):
        try:
            ref_svg = self.get_svg(reference)
            pred_svg = self.get_svg(predictions)

            ref: bytes = svg2png(bytestring=ref_svg)
            pred: bytes = svg2png(bytestring=pred_svg)

            ref_io = BytesIO(ref)
            ref_img = io.imread(ref_io)
            ref_io.close()

            pred_io = BytesIO(pred)
            pred_img = io.imread(pred_io)
            pred_io.close()

            metric_res = ssim(ref_img, pred_img, channel_axis=2)
            return metric_res

        except Exception as e:
            logger.error(f"Error in SVGSSIMMetric: {e}")
            return 0

    @override
    def compute(self, reference, predictions):
        metric_res = 0

        if isinstance(reference, list):
            results = [
                self._compute(ref, pred) for ref, pred in zip(reference, predictions)
            ]
            results = np.array(results)
            metric_res = np.mean(results)
        else:
            metric_res = self._compute(reference, predictions)

        return {self.metric_name: metric_res}
