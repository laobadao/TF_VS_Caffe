
# Tensorflow code

1. 先 clip 2. 后 decode

## clip_to_window()

```
# clip_window:', array([   0,    0,  600, 1002]))
clip_window = np.array([0, 0, image_shape[1], image_shape[2]])

# 'before clip anchors:', array([-45.254834, -22.627417,  45.254834,  22.627417])
anchors = clip_to_window(anchors, clip_window)
#'after clip anchors:', array([ 0.      ,  0.      , 45.254834, 22.627417])

```

```
def clip_to_window(boxlist, window, filter_nonoverlapping=True):

    y_min, x_min, y_max, x_max = np.split(

        boxlist, 4, axis=1)
    win_y_min, win_x_min, win_y_max, win_x_max = np.split(window, 4)

    y_min_clipped = np.maximum(np.minimum(y_min, win_y_max), win_y_min)
    y_max_clipped = np.maximum(np.minimum(y_max, win_y_max), win_y_min)
    x_min_clipped = np.maximum(np.minimum(x_min, win_x_max), win_x_min)
    x_max_clipped = np.maximum(np.minimum(x_max, win_x_max), win_x_min)

    clipped = np.concatenate([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], 1)

    if filter_nonoverlapping:
        areas = _area(clipped)
        greater = np.greater(areas, 0.0)
        where = np.where(greater)
        reshaped = np.reshape(where, [-1])
        nonzero_area_indices = reshaped
        clipped = _gather(clipped, nonzero_area_indices)

    return clipped

def _area(boxlist):
    """Computes area of boxes.

    Args:
      boxlist: BoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing box areas.
    """

    y_min, x_min, y_max, x_max = np.split(
        boxlist, 4, axis=1)

    return np.squeeze((y_max - y_min) * (x_max - x_min), [1])

def _gather(boxlist, indices):
    if indices.ndim != 1:
        raise ValueError('indices should have rank 1')
    if indices.dtype != np.int32 and indices.dtype != np.int64:
        raise ValueError('indices should be an int32 / int64 tensor')

    gathered_result = np.take(boxlist, indices, axis=0)

    return gathered_result
```
