"""Generates grid anchors on the fly as used in Faster RCNN.

Generates grid anchors on the fly as described in:
"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
"""

import numpy as np


def generate_anchors(scales=(0.5, 1.0, 2.0),
                     aspect_ratios=(0.5, 1.0, 2.0),
                     base_anchor_size=None,
                     anchor_stride=None,
                     anchor_offset=None,
                     feature_map_shape_list=None):
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

    if base_anchor_size is None:
        base_anchor_size = [256, 256]
    base_anchor_size = base_anchor_size
    if anchor_stride is None:
        anchor_stride = [16, 16]
    anchor_stride = anchor_stride
    if anchor_offset is None:
        anchor_offset = [0, 0]
    anchor_offset = anchor_offset

    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == 1):
        raise ValueError('feature_map_shape_list must be a list of length 1.')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
        raise ValueError('feature_map_shape_list must be a list of pairs.')
    grid_height, grid_width = feature_map_shape_list[0]

    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    scales_grid = np.reshape(scales_grid, [-1])
    aspect_ratios_grid = np.reshape(aspect_ratios_grid, [-1])
    anchors = _tile_anchors(grid_height,
                            grid_width,
                            scales_grid,
                            aspect_ratios_grid,
                            base_anchor_size,
                            anchor_stride,
                            anchor_offset)
    return anchors


def _tile_anchors(grid_height,
                  grid_width,
                  scales,
                  aspect_ratios,
                  base_anchor_size,
                  anchor_stride,
                  anchor_offset):
    ratio_sqrts = np.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    y_centers = np.array([float(i) for i in range(grid_height)])
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = np.array([float(i) for i in range(grid_width)])
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = _meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = _meshgrid(heights, y_centers)

    bbox_centers = np.stack((y_centers_grid, x_centers_grid), axis=3)
    bbox_sizes = np.stack((heights_grid, widths_grid), axis=3)
    bbox_centers = np.reshape(bbox_centers, (-1, 2))
    bbox_sizes = np.reshape(bbox_sizes, (-1, 2))

    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    return bbox_corners


def _expanded_shape_np(orig_shape, start_dim, num_dims):
    start_dim = np.expand_dims(start_dim, 0)  # scalar to rank-1
    before = orig_shape[0:start_dim[0]]
    shape_p = np.reshape(num_dims, [1])
    add_shape = np.ones(shape=shape_p, dtype=np.int32)
    after = orig_shape[start_dim[0]:]
    new_shape = np.concatenate((before, add_shape, after), 0)
    return new_shape


def _meshgrid(x, y):
    x_exp_shape = _expanded_shape_np(np.array(x.shape), 0, y.ndim)
    y_exp_shape = _expanded_shape_np(np.array(y.shape), y.ndim, x.ndim)
    xgrid = np.tile(np.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = np.tile(np.reshape(y, y_exp_shape), x_exp_shape)
    y_shape = np.array(y.shape, dtype=np.int32)
    x_shape = np.array(x.shape, dtype=np.int32)
    new_shape = np.concatenate((y_shape, x_shape))
    xgrid.reshape(new_shape)
    ygrid.reshape(new_shape)
    return xgrid, ygrid


def _center_size_bbox_to_corners_bbox(centers, sizes):
    return np.concatenate([centers - .5 * sizes, centers + .5 * sizes], 1)


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
