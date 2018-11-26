import numpy as np


def combined_static_and_dynamic_shape(np_array):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      np_array: A np_array of any type.

    Returns:
      A list of size np_array.shape.ndims containing integers or a scalar np_array.
    """
    static_np_array_shape = np_array.shape
    combined_shape = []
    for index, dim in enumerate(static_np_array_shape):
        if dim is not None:
            combined_shape.append(dim)
    return combined_shape


def static_or_dynamic_map_fn(fn, elems, name="sss", num=7):
    arg_tuples = zip(*[np.split(elem, elem.shape[0]) for elem in elems])

    if isinstance(elems, list):
        outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
    else:
        outputs = [fn(elems)]

    if all([isinstance(output, np.ndarray) for output in outputs]):
        return np.stack(outputs)
    return [np.stack(output_tuple) for output_tuple in zip(*outputs)]


def pad_tensor(t, length):
    t_d0 = t.shape[0]
    pad_d0 = np.expand_dims(length - t_d0, 0)

    if t.ndim > 1:
        pad_shape = np.concatenate([pad_d0, t.shape[1:]], 0)
    else:
        pad_shape = np.expand_dims(length - t_d0, 0)

    padded_t = np.concatenate([t, np.zeros(pad_shape, dtype=t.dtype)], 0)
    return padded_t


def clip_tensor(t, length):
    clipped_t = np.take(t, range(length), axis=0)
    return clipped_t


def pad_or_clip_tensor(t, length, name="www"):
    # if t.shape[0] > length:
    #     processed_t = clip_tensor(t, length)
    # else:
    #     processed_t = pad_tensor(t, length)
    processed_t = t
    return processed_t
