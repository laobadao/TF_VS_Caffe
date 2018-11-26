
"""A function to build an object detection box coder from configuration."""
from ..np_utils import faster_rcnn_box_coder
import config

def build(box_coder_config):
    """Builds a box coder object based on the box coder config.

    Args:
      box_coder_config: A box_coder.proto object containing the config for the
        desired box coder.

    Returns:
      BoxCoder based on the config.

    Raises:
      ValueError: On empty box coder proto.
    """
    y_scale = config.cfg.POSTPROCESSOR.Y_SCALE
    x_scale = config.cfg.POSTPROCESSOR.X_SCALE
    height_scale = config.cfg.POSTPROCESSOR.HEIGHT_SCALE
    width_scale = config.cfg.POSTPROCESSOR.WIDTH_SCALE

    if box_coder_config == 'faster_rcnn_box_coder':
        return faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[y_scale, x_scale, height_scale, width_scale])
    raise ValueError('Empty box coder.')
