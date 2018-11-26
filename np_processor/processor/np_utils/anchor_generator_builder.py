"""A function to build an object detection anchor generator from config."""

from ..np_utils import multiple_grid_anchor_generator, grid_anchor_generator, multiscale_grid_anchor_generator
import config

def build():
    """Builds an anchor generator based on the config.

    Args:
      anchor_generator_config: An anchor_generator.proto object containing the
        config for the desired anchor generator.

    Returns:
      Anchor generator based on the config.

    Raises:
      ValueError: On empty anchor generator proto.
    """
    anchor_generator_config = config.cfg.POSTPROCESSOR.ANCHOR_GENERATOR

    if anchor_generator_config == 'ssd_anchor_generator':

        anchor_strides = None
        anchor_offsets = None
        num_layers = config.cfg.POSTPROCESSOR.NUM_LAYERS
        min_scale = config.cfg.POSTPROCESSOR.MIN_SCALE
        max_scale = config.cfg.POSTPROCESSOR.MAX_SCALE
        aspect_ratios = config.cfg.POSTPROCESSOR.ASPECT_RATIOS

        try:
            reduce_boxes_in_lowest_layer = config.cfg.POSTPROCESSOR.REDUCE_BOXES_IN_LOWEST_LAYER
        except:
            reduce_boxes_in_lowest_layer = True

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
            reduce_boxes_in_lowest_layer=reduce_boxes_in_lowest_layer)

    elif anchor_generator_config == "grid_anchor_generator":

        # SCALES = [0.25, 0.5, 1.0, 2.0]
        # __C.POSTPROCESSOR.ASPECT_RATIOS = [0.5, 1.0, 2.0]
        scales = config.cfg.POSTPROCESSOR.SCALES
        aspect_ratios = config.cfg.POSTPROCESSOR.ASPECT_RATIOS
        height_stride = config.cfg.POSTPROCESSOR.HEIGHT_STRIDE
        width_stride = config.cfg.POSTPROCESSOR.WIDTH_STRIDE
        print("grid_anchor_generator scales:", scales)
        return grid_anchor_generator.GridAnchorGenerator(
            scales=[scale for scale in scales],
            aspect_ratios=[aspect_ratio
                           for aspect_ratio
                           in aspect_ratios],
            base_anchor_size=None,
            anchor_stride=[height_stride,
                           width_stride],
            anchor_offset=None)

    elif anchor_generator_config == "multiscale_anchor_generator":
        print("============== multiscale_anchor_generator ==============")
        min_level = config.cfg.POSTPROCESSOR.MIN_LEVEL
        max_level = config.cfg.POSTPROCESSOR.MAX_LEVEL
        anchor_scale = config.cfg.POSTPROCESSOR.ANCHOR_SCALE
        aspect_ratios = config.cfg.POSTPROCESSOR.ASPECT_RATIOS
        scales_per_octave = config.cfg.POSTPROCESSOR.SCALES_PER_OCTAVE

        return multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator(
            min_level,
            max_level,
            anchor_scale,
            [float(aspect_ratio) for aspect_ratio in aspect_ratios],
            scales_per_octave)

    else:
        raise ValueError('Empty anchor generator.')
