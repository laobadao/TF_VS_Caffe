import numpy as np

from ..np_utils import box_list, box_list_ops, shape_utils


def expanded_shape_np(orig_shape, start_dim, num_dims):
    start_dim = np.expand_dims(start_dim, 0)  # scalar to rank-1
    before = orig_shape[0:start_dim[0]]
    shape_p = np.reshape(num_dims, [1])
    add_shape = np.ones(shape=shape_p, dtype=np.int32)
    after = orig_shape[start_dim[0]:]
    new_shape = np.concatenate((before, add_shape, after), 0)
    return new_shape


def meshgrid(x, y):
    x_exp_shape = expanded_shape_np(np.array(x.shape), 0, y.ndim)
    y_exp_shape = expanded_shape_np(np.array(y.shape), y.ndim, x.ndim)
    xgrid = np.tile(np.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = np.tile(np.reshape(y, y_exp_shape), x_exp_shape)
    y_shape = np.array(y.shape, dtype=np.int32)
    x_shape = np.array(x.shape, dtype=np.int32)
    new_shape = np.concatenate((y_shape, x_shape))
    xgrid.reshape(new_shape)
    ygrid.reshape(new_shape)
    return xgrid, ygrid


def normalized_to_image_coordinates(normalized_boxes, image_shape, parallel_iterations=32):

    def _to_absolute_coordinates(normalized_boxes):
        normalized_boxes = normalized_boxes

        return box_list_ops.to_absolute_coordinates(
            box_list.BoxList(normalized_boxes),
            image_shape[1], image_shape[2]).get()

    absolute_boxes = shape_utils.static_or_dynamic_map_fn(
        _to_absolute_coordinates,
        elems=(normalized_boxes))

    return absolute_boxes


def sigmoid(value):
    value = np.float64(value)
    return 1.0 / (1 + np.exp(-value))

def softmax(z):
    input_shape = z.shape
    if z.ndim == 4:
        z = np.reshape(z, [-1, z.shape[-1]])
    if len(z.shape) == 3 and z.shape[0] == 1:
        z = z[0]
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    # e_x = np.exp(z)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    result = e_x / div
    if len(input_shape) == 3:
        result = np.reshape(result, input_shape)
    if len(input_shape) == 4 and input_shape[0] == 1:
        result = np.reshape(result, input_shape)
    return result
