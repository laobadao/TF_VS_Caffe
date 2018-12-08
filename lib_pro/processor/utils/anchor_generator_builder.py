"""A function to build an object detection anchor generator from config."""

from ..utils import multiple_grid_anchor_generator, grid_anchor_generator


def build(anchor_generator_config, scales=None, aspect_ratios=None, anchor_stride=None):
    """Builds an anchor generator based on the config.

    Args:
      anchor_generator_config: An anchor_generator.proto object containing the
        config for the desired anchor generator.

    Returns:
      Anchor generator based on the config.

    Raises:
      ValueError: On empty anchor generator proto.

    """

    if anchor_generator_config == 'ssd_anchor_generator':
        anchor_strides = None
        anchor_offsets = None
        num_layers = 6
        min_scale = 0.2
        max_scale = 0.95
        aspect_ratios = (1.0, 2.0, 0.5, 3.0, 0.3333)
        return multiple_grid_anchor_generator.create_ssd_anchors(
            num_layers=num_layers,
            min_scale=min_scale,
            max_scale=max_scale,
            scales=None,
            aspect_ratios=aspect_ratios,
            interpolated_scale_aspect_ratio=1.0,
            base_anchor_size=None,
            anchor_strides=anchor_strides,
            anchor_offsets=anchor_offsets,
            reduce_boxes_in_lowest_layer=True)
    elif anchor_generator_config == "grid_anchor_generator":

        print("anchor_stride:", anchor_stride)
        height_stride = anchor_stride[0]
        width_stride = anchor_stride[1]
        return grid_anchor_generator.GridAnchorGenerator(
            scales=[float(scale) for scale in scales],
            aspect_ratios=[float(aspect_ratio)
                           for aspect_ratio
                           in aspect_ratios],
            base_anchor_size=None,
            anchor_stride=[height_stride,
                           width_stride],
            anchor_offset=None)

    else:
        raise ValueError('Empty anchor generator.')
