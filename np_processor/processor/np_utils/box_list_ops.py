"""Bounding Box List operations.

Example box operations that are supported:
  * areas: compute bounding box areas
  * iou: pairwise intersection-over-union scores
  * sq_dist: pairwise distances between bounding boxes

Whenever box_list_ops functions output a BoxList, the fields of the incoming
BoxList are retained unless documented otherwise.
"""
import numpy as np

from ..np_utils import box_list, shape_utils


class SortOrder(object):
    """Enum class for sort order.

    Attributes:
      ascend: ascend order.
      descend: descend order.
    """
    ascend = 1
    descend = 2


def filter_greater_than(boxlist, thresh, num=0, class_idx=800):
    if not isinstance(boxlist, box_list.BoxList):
        raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
        raise ValueError('input boxlist must have \'scores\' field')

    scores = boxlist.get_field('scores')

    if scores.ndim > 2:
        raise ValueError('Scores should have rank 1 or 2')
    if scores.ndim == 2 and scores.shape[1] != 1:
        raise ValueError('Scores should have rank 1 or have shape '
                         'consistent with [None, 1]')

    high_score_indices = np.reshape(np.where(np.greater(scores, thresh)), [-1])

    return gather(boxlist, high_score_indices, num=num, class_idx=class_idx)


def pad_or_clip_box_list(boxlist, num_boxes, num=0):
    """Pads or clips all fields of a BoxList.

    Args:
      boxlist: A BoxList with arbitrary of number of boxes.
      num_boxes: First num_boxes in boxlist are kept.
        The fields are zero-padded if num_boxes is bigger than the
        actual number of boxes.
      scope: name scope.

    Returns:
      BoxList with all fields padded or clipped.
    """

    subboxlist = box_list.BoxList(shape_utils.pad_or_clip_tensor(
        boxlist.get(), num_boxes))

    for field in boxlist.get_extra_fields():
        subfield = shape_utils.pad_or_clip_tensor(
            boxlist.get_field(field), num_boxes)
        subboxlist.add_field(field, subfield)

    return subboxlist


def gather(boxlist, indices, fields=None, name=None, num=0, class_idx=900):
    if indices.ndim != 1:
        raise ValueError('indices should have rank 1')
    if indices.dtype != np.int32 and indices.dtype != np.int64:
        raise ValueError('indices should be an int32 / int64 tensor')
    if fields is None:
        fields = boxlist.get_extra_fields()

    gathered_result = np.take(boxlist.get(), indices, axis=0)
    subboxlist = box_list.BoxList(gathered_result)

    for field in fields:

        if not boxlist.has_field(field):
            raise ValueError('boxlist must contain all specified fields')

        if boxlist.get_field(field).ndim == 3:
            subboxlist_result = np.zeros(shape=(boxlist.get_field(field).shape[0], 0, 0))
        else:
            subboxlist_result = np.take(boxlist.get_field(field), indices, axis=0)
        subboxlist.add_field(field, subboxlist_result)

    return subboxlist

def change_coordinate_frame(boxlist, window):

    win_height = window[2] - window[0]
    win_width = window[3] - window[1]

    boxlist_new = scale(box_list.BoxList(
        boxlist.get() - [window[0], window[1], window[0], window[1]]),
        1.0 / win_height, 1.0 / win_width)

    boxlist_new = _copy_extra_fields(boxlist_new, boxlist)

    return boxlist_new

def intersection(boxlist1, boxlist2):
    y_min1, x_min1, y_max1, x_max1 = np.split(
        boxlist1.get(), 4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = np.split(
        boxlist2.get(), 4, axis=1)
    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxlist1, boxlist2):

    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
            np.expand_dims(areas1, 1) + np.expand_dims(areas2, 0) - intersections)

    iou_result = np.where(
        np.equal(intersections, 0.0),
        np.zeros_like(intersections), np.divide(intersections, unions))

    return iou_result


def area(boxlist):
    """Computes area of boxes.

    Args:
      boxlist: BoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing box areas.
    """

    y_min, x_min, y_max, x_max = np.split(
        boxlist.get(), 4, axis=1)

    return np.squeeze((y_max - y_min) * (x_max - x_min), [1])


def to_normalized_coordinates(boxlist, height, width):

    height = float(height)
    width = float(width)
    return scale(boxlist, 1 / height, 1 / width)

def scale(boxlist, y_scale, x_scale):

    y_scale = float(y_scale)
    x_scale = float(x_scale)

    y_min, x_min, y_max, x_max = np.split(
        boxlist.get(), 4, axis=1)

    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max

    concatenated = np.concatenate([y_min, x_min, y_max, x_max], 1)
    scaled_boxlist = box_list.BoxList(concatenated)

    return _copy_extra_fields(scaled_boxlist, boxlist)

def clip_to_window(boxlist, window, filter_nonoverlapping=True):

    y_min, x_min, y_max, x_max = np.split(

        boxlist.get(), 4, axis=1)
    win_y_min, win_x_min, win_y_max, win_x_max = np.split(window, 4)

    y_min_clipped = np.maximum(np.minimum(y_min, win_y_max), win_y_min)
    y_max_clipped = np.maximum(np.minimum(y_max, win_y_max), win_y_min)
    x_min_clipped = np.maximum(np.minimum(x_min, win_x_max), win_x_min)
    x_max_clipped = np.maximum(np.minimum(x_max, win_x_max), win_x_min)

    clipped = box_list.BoxList(
        np.concatenate([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], 1))

    clipped = _copy_extra_fields(clipped, boxlist)

    if filter_nonoverlapping:
        areas = area(clipped)
        greater = np.greater(areas, 0.0)
        where = np.where(greater)
        reshaped = np.reshape(where, [-1])
        nonzero_area_indices = reshaped
        clipped = gather(clipped, nonzero_area_indices)

    return clipped


def concatenate(boxlists, fields=None, name=None):
    if not isinstance(boxlists, list):
        raise ValueError('boxlists should be a list')
    if not boxlists:
        raise ValueError('boxlists should have nonzero length')
    for boxlist in boxlists:
        if not isinstance(boxlist, box_list.BoxList):
            raise ValueError('all elements of boxlists should be BoxList objects')

    boxlisted = [boxlist.get() for boxlist in boxlists]
    concated = np.concatenate(boxlisted, 0)
    concatenated = box_list.BoxList(concated)
    if fields is None:
        fields = boxlists[0].get_extra_fields(name="selected_boxes")

    for field in fields:
        first_field_shape = list(boxlists[0].get_field(field).shape)
        first_field_shape[0] = -1
        if None in first_field_shape:
            raise ValueError('field %s must have fully defined shape except for the'
                             ' 0th dimension.' % field)
        for boxlist in boxlists:
            if not boxlist.has_field(field):
                raise ValueError('boxlist must contain all requested fields')
            field_shape = list(boxlist.get_field(field).shape)
            field_shape[0] = -1
            if field_shape != first_field_shape:
                raise ValueError('field %s must have same shape for all boxlists '
                                 'except for the 0th dimension.' % field)

        concatenated_field = np.concatenate(
            [boxlist.get_field(field) for boxlist in boxlists], 0)

        concatenated.add_field(field, concatenated_field)

    return concatenated


def sort_by_field(boxlist, field, order=SortOrder.descend):
    if order != SortOrder.descend and order != SortOrder.ascend:
        raise ValueError('Invalid sort order')

    field_to_sort = boxlist.get_field(field)
    if field_to_sort.ndim != 1:
        raise ValueError('Field should have rank 1')

    num_boxes = boxlist.num_boxes()

    top_k_result = np.argpartition(field_to_sort, np.array(range(num_boxes)))[::-1]
    return gather(boxlist, top_k_result)


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):

    for field in boxlist_to_copy_from.get_extra_fields():
        print("field:", field)
        boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))

    return boxlist_to_copy_to

def to_absolute_coordinates(boxlist,
                            height,
                            width,
                            maximum_normalized_coordinate=1.1):

    return scale(boxlist, height, width)
