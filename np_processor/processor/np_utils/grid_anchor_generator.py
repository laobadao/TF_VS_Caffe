"""Generates grid anchors on the fly as used in Faster RCNN.

Generates grid anchors on the fly as described in:
"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
"""

from ..np_utils import box_list, anchor_generator, ops
import numpy as np


class GridAnchorGenerator(anchor_generator.AnchorGenerator):
    """Generates a grid of anchors at given scales and aspect ratios."""

    def __init__(self,
                 scales=(0.5, 1.0, 2.0),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 base_anchor_size=None,
                 anchor_stride=None,
                 anchor_offset=None):
        """Constructs a GridAnchorGenerator.

        Args:
          scales: a list of (float) scales, default=(0.5, 1.0, 2.0)
          aspect_ratios: a list of (float) aspect ratios, default=(0.5, 1.0, 2.0)
          base_anchor_size: base anchor size as height, width (
                            (length-2 float32 list or tensor, default=[256, 256])
          anchor_stride: difference in centers between base anchors for adjacent
                         grid positions (length-2 float32 list or tensor,
                         default=[16, 16])
          anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                         upper left element of the grid, this should be zero for
                         feature networks with only VALID padding and even receptive
                         field size, but may need additional calculation if other
                         padding is used (length-2 float32 list or tensor,
                         default=[0, 0])
        """
        # Handle argument defaults
        if base_anchor_size is None:
            base_anchor_size = [256, 256]
        base_anchor_size = base_anchor_size
        if anchor_stride is None:
            anchor_stride = [16, 16]
        anchor_stride = anchor_stride
        if anchor_offset is None:
            anchor_offset = [0, 0]
        anchor_offset = anchor_offset

        self._scales = scales
        self._aspect_ratios = aspect_ratios
        self._base_anchor_size = base_anchor_size
        self._anchor_stride = anchor_stride
        self._anchor_offset = anchor_offset

    def num_anchors_per_location(self):
        """Returns the number of anchors per spatial location.

        Returns:
          a list of integers, one for each expected feature map to be passed to
          the `generate` function.
        """
        return [len(self._scales) * len(self._aspect_ratios)]

    def _generate(self, feature_map_shape_list):
        """Generates a collection of bounding boxes to be used as anchors.

        Args:
          feature_map_shape_list: list of pairs of convnet layer resolutions in the
            format [(height_0, width_0)].  For example, setting
            feature_map_shape_list=[(8, 8)] asks for anchors that correspond
            to an 8x8 layer.  For this anchor generator, only lists of length 1 are
            allowed.

        Returns:
          boxes_list: a list of BoxLists each holding anchor boxes corresponding to
            the input feature map shapes.

        Raises:
          ValueError: if feature_map_shape_list, box_specs_list do not have the same
            length.
          ValueError: if feature_map_shape_list does not consist of pairs of
            integers
        """
        print("feature_map_shape_list:", feature_map_shape_list)
        if not (isinstance(feature_map_shape_list, list)
                and len(feature_map_shape_list) == 1):
            raise ValueError('feature_map_shape_list must be a list of length 1.')
        if not all([isinstance(list_item, tuple) and len(list_item) == 2
                    for list_item in feature_map_shape_list]):
            raise ValueError('feature_map_shape_list must be a list of pairs.')
        grid_height, grid_width = feature_map_shape_list[0] # 38*63

        scales_grid, aspect_ratios_grid = np.meshgrid(self._scales, self._aspect_ratios)
        # scales_grid 图像尺寸 缩放比 256 为 base_size
        # [[0.25 0.5  1.   2.  ]
        #  [0.25 0.5  1.   2.  ]
        #  [0.25 0.5  1.   2.  ]]
        # aspect_ratios_grid 宽高比
        # [[0.5 0.5 0.5 0.5]
        #  [1.  1.  1.  1. ]
        #  [2.  2.  2.  2. ]]
        scales_grid = np.reshape(scales_grid, [-1])
        aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])

        # scales_grid: [0.25 0.5  1.   2.   0.25 0.5  1.   2.   0.25 0.5  1.   2.  ]
        # aspect_ratios_grid: [0.5 0.5 0.5 0.5 1.  1.  1.  1.  2.  2.  2.  2. ]

        anchors = tile_anchors(grid_height,
                               grid_width,
                               scales_grid,
                               aspect_ratios_grid,
                               self._base_anchor_size,
                               self._anchor_stride,
                               self._anchor_offset)

        num_anchors = anchors.num_boxes_static()
        if num_anchors is None:
            num_anchors = anchors.num_boxes()
        anchor_indices = np.zeros([num_anchors])
        anchors.add_field('feature_map_index', anchor_indices)
        return [anchors]


def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):

    ratio_sqrts = np.sqrt(aspect_ratios)
    # sqrt aspect_ratios_grid: [0.5 0.5 0.5 0.5 1.  1.  1.  1.  2.  2.  2.  2. ]
    # ratio_sqrts:
    # [0.70710678 0.70710678 0.70710678 0.70710678 1.         1.
    #  1.         1.         1.41421356 1.41421356 1.41421356 1.41421356]

    # scales_grid: [0.25 0.5  1.   2.   0.25 0.5  1.   2.   0.25 0.5  1.   2.  ]
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    # heights: [ 90.50966799 181.01933598 362.03867197 724.07734394  64.
    #  128.         256.         512.          45.254834    90.50966799
    #  181.01933598 362.03867197]

    # widths: [ 45.254834    90.50966799 181.01933598 362.03867197  64.
    #  128.         256.         512.          90.50966799 181.01933598
    #  362.03867197 724.07734394]

    # Get a grid of box centers

    # anchor_stride: [16, 16]
    # anchor_offset: [0, 0]

    y_centers = np.array([float(i) for i in range(grid_height)])

    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]


    # y_centers: [  0.  16.  32.  48.  64.  80.  96. 112. 128. 144. 160. 176. 192. 208.
    #  224. 240. 256. 272. 288. 304. 320. 336. 352. 368. 384. 400. 416. 432.
    #  448. 464. 480. 496. 512. 528. 544. 560. 576. 592.] 38 个
    x_centers = np.array([float(i) for i in range(grid_width)])
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]

    # x_centers: [  0.  16.  32.  48.  64.  80.  96. 112. 128. 144. 160. 176. 192. 208.
    #  224. 240. 256. 272. 288. 304. 320. 336. 352. 368. 384. 400. 416. 432.
    #  448. 464. 480. 496. 512. 528. 544. 560. 576. 592.
    # 608. 624. 640. 656.
    #  672. 688. 704. 720. 736. 752. 768. 784. 800. 816. 832. 848. 864. 880.
    #  896. 912. 928. 944. 960. 976. 992.]  63 个

    x_centers, y_centers = np.meshgrid(x_centers, y_centers)
    # x_centers: (38, 63)
    # y_centers: (38, 63)
    widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)

    bbox_centers = np.stack((y_centers_grid, x_centers_grid), axis=3)
    bbox_sizes = np.stack((heights_grid, widths_grid), axis=3)
    bbox_centers = np.reshape(bbox_centers, (-1, 2))
    bbox_sizes = np.reshape(bbox_sizes, (-1, 2))

    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    # bbox_centers: (28728, 2)
    # bbox_sizes: (28728, 2)
    # bbox_corners: (28728, 4)

    # bbox_corners [0]: [-45.254834 -22.627417  45.254834  22.627417]
    # bbox_corners [1]: [-90.50966799 -45.254834    90.50966799  45.254834  ]

    return box_list.BoxList(bbox_corners)


def _center_size_bbox_to_corners_bbox(centers, sizes):
    return np.concatenate([centers - .5 * sizes, centers + .5 * sizes], 1)
