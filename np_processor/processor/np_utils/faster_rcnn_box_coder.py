
"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

  See http://arxiv.org/abs/1506.01497 for details.
"""

from ..np_utils import box_coder, box_list

import numpy as np

EPSILON = 1e-8


class FasterRcnnBoxCoder(box_coder.BoxCoder):
    """Faster RCNN box coder."""

    def __init__(self, scale_factors=None):
        """Constructor for FasterRcnnBoxCoder.

        Args:
          scale_factors: List of 4 positive scalars to scale ty, tx, th and tw.
            If set to None, does not perform scaling. For Faster RCNN,
            the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].
        """
        if scale_factors:
            assert len(scale_factors) == 4
            for scalar in scale_factors:
                assert scalar > 0
        self._scale_factors = scale_factors

    @property
    def code_size(self):
        return 4

    def _encode(self, boxes, anchors):
        print("============== _encode =============")
        """Encode a box collection with respect to anchor collection.

        Args:
          boxes: BoxList holding N boxes to be encoded.
          anchors: BoxList of anchors.

        Returns:
          a tensor representing N anchor-encoded boxes of the format
          [ty, tx, th, tw].
        """
        # Convert anchors to the center coordinate representation.
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
        ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
        # Avoid NaN in division and log below.
        # EPSILON = 1e-8
        ha += EPSILON
        wa += EPSILON
        h += EPSILON
        w += EPSILON

        tx = (xcenter - xcenter_a) / wa
        ty = (ycenter - ycenter_a) / ha
        tw = np.log(w / wa)
        th = np.log(h / ha)
        # Scales location targets as used in paper for joint training.
        if self._scale_factors:
            ty *= self._scale_factors[0]
            tx *= self._scale_factors[1]
            th *= self._scale_factors[2]
            tw *= self._scale_factors[3]
        return np.transpose(np.stack([ty, tx, th, tw]))

    def _decode(self, rel_codes, anchors):
        print("============== _decode =============")
        """Decode relative codes to boxes.

        Args:
          rel_codes: a tensor representing N anchor-encoded boxes.
          anchors: BoxList of anchors.

        Returns:
          boxes: BoxList holding N bounding boxes.
        """
        # rel_codes: (28728, 4)
        # print("rel_codes:", rel_codes.shape)
        # print("rel_codes[0]:", rel_codes[0])
        # print("anchors[0]:", anchors.get()[0])
        # 将 左上角，右下角坐标，转换为 中心点和宽高
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
        
        # print("rel_codes:", rel_codes[0])
        tras_rel_codes = np.transpose(rel_codes)
        # tras_rel_codes: (4, 28728)
        ty, tx, th, tw = np.split(tras_rel_codes, 4)
        ty = ty.flatten()
        tx = tx.flatten()
        th = th.flatten()
        tw = tw.flatten()
        # self._scale_factors: [10.0, 10.0, 5.0, 5.0]

        if self._scale_factors:
            ty /= self._scale_factors[0]
            tx /= self._scale_factors[1]
            th /= self._scale_factors[2]
            tw /= self._scale_factors[3]
            
        w = np.exp(tw) * wa
        h = np.exp(th) * ha

        # print("w {} , wa {}".format(w[0], wa[0]))
        # print("h {} , ha {}".format(h[0], ha[0]))

        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a

        # print("ycenter {} , ycenter_a {}".format(ycenter[0], ycenter_a[0]))
        # print("xcenter {} , xcenter_a {}".format(xcenter[0], xcenter_a[0]))

        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        decode_proposal = np.transpose(np.stack([ymin, xmin, ymax, xmax]))
        print("decode_proposal:", decode_proposal.shape)
        print("================= decode_proposal[0]:", decode_proposal[0])
        return box_list.BoxList(decode_proposal)
