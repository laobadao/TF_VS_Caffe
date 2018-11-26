
"""Post-processing operations on detected boxes."""

import numpy as np

from ..np_utils import box_list, box_list_ops, shape_utils
from ..np_utils import standard_fields as fields
import util.bbox as bbox


def multiclass_non_max_suppression(boxes,
                                   scores,
                                   score_thresh,
                                   iou_thresh,
                                   max_size_per_class,
                                   max_total_size=0,
                                   clip_window=None,
                                   change_coordinate_frame=False,
                                   masks=None,
                                   boundaries=None,
                                   additional_fields=None, num=0):

    if not 0 <= iou_thresh <= 1.0:
        raise ValueError('iou_thresh must be between 0 and 1')
    if scores.ndim != 2:
        raise ValueError('scores field must be of rank 2')
    if scores.shape[1] is None:
        raise ValueError('scores must have statically defined second '
                         'dimension')
    if boxes.ndim != 3:
        raise ValueError('boxes must be of rank 3.')
    if not (boxes.shape[1] == scores.shape[1] or
            boxes.shape[1] == 1):
        raise ValueError('second dimension of boxes must be either 1 or equal '
                         'to the second dimension of scores')
    if boxes.shape[2] != 4:
        raise ValueError('last dimension of boxes must be of size 4.')
    if change_coordinate_frame and clip_window is None:
        raise ValueError('if change_coordinate_frame is True, then a clip_window'
                         'must be specified.')

    num_boxes = boxes.shape[0]
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]

    if np.equal(num_boxes, num_scores):
        length_assert = True
    else:
        raise ValueError('Incorrect scores field length: actual vs expected.', num_scores, num_boxes)

    selected_boxes_list = []

    per_class_boxes_list = np.split(boxes, boxes.shape[1], axis=1)
    if masks is not None:
        per_class_masks_list = masks

    boxes_ids = (range(num_classes) if len(per_class_boxes_list) > 1
                 else [0] * num_classes)

    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):

        per_class_boxes = per_class_boxes_list[boxes_idx]

        if per_class_boxes.ndim == 3 and per_class_boxes.shape[1] == 1:
            per_class_boxes = per_class_boxes.reshape((per_class_boxes.shape[0], per_class_boxes.shape[2]))

        boxlist_and_class_scores = box_list.BoxList(per_class_boxes)

        sliced_scores = scores[:, class_idx]

        class_scores = np.reshape(sliced_scores, sliced_scores.shape[0])

        boxlist_and_class_scores.add_field(fields.BoxListFields.scores,
                                           class_scores)

        if masks is not None:
            if boxes_idx >= len(per_class_masks_list):
                per_class_masks = np.zeros(shape=(1, 0, 0))
            else:
                per_class_masks = per_class_masks_list[boxes_idx]

            boxlist_and_class_scores.add_field(fields.BoxListFields.masks,
                                               per_class_masks)

        if additional_fields is not None:
            for key, tensor in additional_fields.items():
                boxlist_and_class_scores.add_field(key, tensor)

        boxlist_filtered = box_list_ops.filter_greater_than(
            boxlist_and_class_scores, score_thresh, num=num, class_idx=class_idx)


        if clip_window is not None:
            boxlist_filtered = box_list_ops.clip_to_window(
                boxlist_filtered, clip_window)

            if change_coordinate_frame:
                boxlist_filtered = box_list_ops.change_coordinate_frame(
                    boxlist_filtered, clip_window)

        max_selection_size = np.minimum(max_size_per_class,
                                        boxlist_filtered.num_boxes())

        selected_indices_np = bbox.apply_nms(boxlist_filtered.get(),
                       boxlist_filtered.get_field(fields.BoxListFields.scores),
                                             iou_thresh, 1, 0, max_selection_size)

        selected_indices_result = np.array(selected_indices_np, dtype=np.int32)
        nms_result = box_list_ops.gather(boxlist_filtered, selected_indices_result)
        zeros = np.zeros_like(
            nms_result.get_field(fields.BoxListFields.scores))
        nms_result.add_field(
            fields.BoxListFields.classes, (zeros + class_idx))
        selected_boxes_list.append(nms_result)

    selected_boxes = box_list_ops.concatenate(selected_boxes_list)

    # argpartition top_k
    sorted_boxes = box_list_ops.sort_by_field(selected_boxes,
                                              fields.BoxListFields.scores)
    if max_total_size:
        max_total_size = np.minimum(max_total_size,
                                    sorted_boxes.num_boxes())

        sorted_boxes = box_list_ops.gather(sorted_boxes, np.array(range(max_total_size)))

    return sorted_boxes


def batch_multiclass_non_max_suppression(boxes,
                                         scores,
                                         score_thresh,
                                         iou_thresh,
                                         max_size_per_class,
                                         max_total_size=0,
                                         clip_window=None,
                                         change_coordinate_frame=False,
                                         num_valid_boxes=None,
                                         masks=None,
                                         additional_fields=None,
                                         parallel_iterations=32,
                                         num=9):

    q = boxes.shape[2]
    num_classes = scores.shape[2]

    if q != 1 and q != num_classes:
        raise ValueError('third dimension of boxes must be either 1 or equal '
                         'to the third dimension of scores')
    if change_coordinate_frame and clip_window is None:
        raise ValueError('if change_coordinate_frame is True, then a clip_window'
                         'must be specified.')
    original_additional_fields = additional_fields

    boxes_shape = boxes.shape
    batch_size = boxes_shape[0]
    num_anchors = boxes_shape[1]

    if batch_size is None:
        batch_size = np.shape(boxes)[0]
    if num_anchors is None:
        num_anchors = np.shape(boxes)[1]

    # If num valid boxes aren't provided, create one and mark all boxes as
    # valid.
    if num_valid_boxes is None:
        num_valid_boxes = np.ones([batch_size], dtype=np.int32) * num_anchors

    # If masks aren't provided, create dummy masks so we can only have one copy
    # of _single_image_nms_fn and discard the dummy masks after map_fn.

    if masks is None:
        masks_shape = np.stack([batch_size, num_anchors, 1, 0, 0])
        masks = np.zeros(masks_shape)

    if clip_window is None:
        clip_window = np.stack([
            np.min(boxes[:, :, :, 0]),
            np.min(boxes[:, :, :, 1]),
            np.max(boxes[:, :, :, 2]),
            np.max(boxes[:, :, :, 3])
        ])

    if clip_window.ndim == 1:
        clip_window = np.tile(np.expand_dims(clip_window, 0), [batch_size, 1])

    if additional_fields is None:
        additional_fields = {}

    def _single_image_nms_fn(args, num=0):

        per_image_boxes = args[0][0]
        per_image_scores = args[1][0]
        per_image_masks = args[2][0]
        per_image_clip_window = args[3][0]

        per_image_additional_fields = {
            key: value
            for key, value in zip(additional_fields, args[4:-1])
        }

        per_image_boxes = np.reshape(per_image_boxes, [-1, q, 4])
        per_image_scores = np.reshape(per_image_scores, [-1, num_classes])

        nmsed_boxlist = multiclass_non_max_suppression(
            per_image_boxes,
            per_image_scores,
            score_thresh,
            iou_thresh,
            max_size_per_class,
            max_total_size,
            clip_window=per_image_clip_window,
            change_coordinate_frame=change_coordinate_frame,
            masks=per_image_masks,
            additional_fields=per_image_additional_fields)

        padded_boxlist = box_list_ops.pad_or_clip_box_list(nmsed_boxlist,
                                                           max_total_size, num=num)
        num_detections = nmsed_boxlist.num_boxes()

        nmsed_boxes = padded_boxlist.get()
        nmsed_scores = padded_boxlist.get_field(fields.BoxListFields.scores)
        nmsed_classes = padded_boxlist.get_field(fields.BoxListFields.classes)
        nmsed_masks = padded_boxlist.get_field(fields.BoxListFields.masks)
        nmsed_additional_fields = [
            padded_boxlist.get_field(key) for key in per_image_additional_fields
        ]
        return ([nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks] +
                nmsed_additional_fields + [num_detections])

    num_additional_fields = 0
    if additional_fields is not None:
        num_additional_fields = len(additional_fields)

    num_nmsed_outputs = 4 + num_additional_fields

    elems = ([boxes, scores, masks, clip_window] +
             list(additional_fields.values()) + [num_valid_boxes])

    batch_outputs = shape_utils.static_or_dynamic_map_fn(
        _single_image_nms_fn,
        elems=elems, name="batch_outputs", num=num)

    batch_nmsed_boxes = batch_outputs[0]
    batch_nmsed_scores = batch_outputs[1]
    batch_nmsed_classes = batch_outputs[2]
    batch_nmsed_masks = batch_outputs[3]

    batch_nmsed_additional_fields = {
        key: value
        for key, value in zip(additional_fields, batch_outputs[4:-1])
    }
    batch_num_detections = batch_outputs[-1]

    if original_additional_fields is None:
        batch_nmsed_additional_fields = None

    return (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes,
            batch_nmsed_masks, batch_nmsed_additional_fields,
            batch_num_detections)
