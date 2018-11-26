from ..np_utils import faster_rcnn_box_coder, mean_stddev_box_coder
from ..np_utils import box_coder as bcoder
from ..np_utils import box_list
from ..np_utils import matcher as mat
from ..np_utils import region_similarity_calculator as sim_calc
from ..np_utils import standard_fields as fields
from ..np_utils import argmax_matcher, bipartite_matcher, shape_utils
import numpy as np


class TargetAssigner(object):
    """Target assigner to compute classification and regression targets."""

    def __init__(self, similarity_calc, matcher, box_coder,
                 negative_class_weight=1.0, unmatched_cls_target=None):

        if not isinstance(similarity_calc, sim_calc.RegionSimilarityCalculator):
            raise ValueError('similarity_calc must be a RegionSimilarityCalculator')
        if not isinstance(matcher, mat.Matcher):
            raise ValueError('matcher must be a Matcher')
        if not isinstance(box_coder, bcoder.BoxCoder):
            raise ValueError('box_coder must be a BoxCoder')
        self._similarity_calc = similarity_calc
        self._matcher = matcher
        self._box_coder = box_coder
        self._negative_class_weight = negative_class_weight

        if unmatched_cls_target is None:
            self._unmatched_cls_target = [0]
        else:
            self._unmatched_cls_target = unmatched_cls_target

    @property
    def box_coder(self):
        return self._box_coder

    def assign(self, anchors, groundtruth_boxes, groundtruth_labels=None,
               groundtruth_weights=None, **params):

        if not isinstance(anchors, box_list.BoxList):
            raise ValueError('anchors must be an BoxList')
        if not isinstance(groundtruth_boxes, box_list.BoxList):
            raise ValueError('groundtruth_boxes must be an BoxList')

        if groundtruth_labels is None:
            groundtruth_labels = np.ones(np.expand_dims(groundtruth_boxes.num_boxes(),
                                                        0))
            groundtruth_labels = np.expand_dims(groundtruth_labels, -1)


        if groundtruth_weights is None:
            num_gt_boxes = groundtruth_boxes.num_boxes_static()
            if not num_gt_boxes:
                num_gt_boxes = groundtruth_boxes.num_boxes()
            groundtruth_weights = np.ones([num_gt_boxes], dtype=np.float32)

        print("=========== _similarity_calc.compare ============")
        match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,
                                                             anchors)

        match = self._matcher.match(match_quality_matrix, **params)
        reg_targets = self._create_regression_targets(anchors,
                                                      groundtruth_boxes,
                                                      match)
        cls_targets = self._create_classification_targets(groundtruth_labels,
                                                          match)
        reg_weights = self._create_regression_weights(match, groundtruth_weights)
        cls_weights = self._create_classification_weights(match,
                                                          groundtruth_weights)

        num_anchors = anchors.num_boxes_static()
        if num_anchors is not None:
            reg_targets = self._reset_target_shape(reg_targets, num_anchors)
            cls_targets = self._reset_target_shape(cls_targets, num_anchors)
            reg_weights = self._reset_target_shape(reg_weights, num_anchors)
            cls_weights = self._reset_target_shape(cls_weights, num_anchors)

        return cls_targets, cls_weights, reg_targets, reg_weights, match

    def _reset_target_shape(self, target, num_anchors):

        target_shape = target.get_shape().as_list()
        target_shape[0] = num_anchors
        target.set_shape(target_shape)
        return target

    def _create_regression_targets(self, anchors, groundtruth_boxes, match):

        matched_gt_boxes = match.gather_based_on_match(
            groundtruth_boxes.get(),
            unmatched_value=np.zeros(4),
            ignored_value=np.zeros(4))
        matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
        if groundtruth_boxes.has_field(fields.BoxListFields.keypoints):
            groundtruth_keypoints = groundtruth_boxes.get_field(
                fields.BoxListFields.keypoints)
            matched_keypoints = match.gather_based_on_match(
                groundtruth_keypoints,
                unmatched_value=np.zeros(groundtruth_keypoints.get_shape()[1:]),
                ignored_value=np.zeros(groundtruth_keypoints.get_shape()[1:]))
            matched_gt_boxlist.add_field(fields.BoxListFields.keypoints,
                                         matched_keypoints)
        matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
        match_results_shape = shape_utils.combined_static_and_dynamic_shape(
            match.match_results)

        # Zero out the unmatched and ignored regression targets.
        unmatched_ignored_reg_targets = np.tile(
            self._default_regression_target(), [match_results_shape[0], 1])
        matched_anchors_mask = match.matched_column_indicator()
        reg_targets = np.where(matched_anchors_mask,
                               matched_reg_targets,
                               unmatched_ignored_reg_targets)
        return reg_targets

    def _default_regression_target(self):

        return [self._box_coder.code_size * [0]]

    def _create_classification_targets(self, groundtruth_labels, match):

        return match.gather_based_on_match(
            groundtruth_labels,
            unmatched_value=self._unmatched_cls_target,
            ignored_value=self._unmatched_cls_target)

    def _create_regression_weights(self, match, groundtruth_weights):

        return match.gather_based_on_match(
            groundtruth_weights, ignored_value=0., unmatched_value=0.)

    def _create_classification_weights(self,
                                       match,
                                       groundtruth_weights):

        return match.gather_based_on_match(
            groundtruth_weights,
            ignored_value=0.,
            unmatched_value=self._negative_class_weight)

    def get_box_coder(self):
        """Get BoxCoder of this TargetAssigner.

        Returns:
          BoxCoder object.
        """
        return self._box_coder


def create_target_assigner(reference, stage=None,
                           negative_class_weight=1.0,
                           unmatched_cls_target=None):
    if reference == 'FasterRCNN' and stage == 'proposal':

        similarity_calc = sim_calc.IouSimilarity()

        matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.7,
                                               unmatched_threshold=0.3,
                                               force_match_for_each_row=True)

        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(
            scale_factors=[10.0, 10.0, 5.0, 5.0])

    else:
        raise ValueError('No valid combination of reference and stage.')

    return TargetAssigner(similarity_calc, matcher, box_coder,
                          negative_class_weight=negative_class_weight,
                          unmatched_cls_target=unmatched_cls_target)


def batch_assign_targets(target_assigner,
                         anchors_batch,
                         gt_box_batch,
                         gt_class_targets_batch,
                         gt_weights_batch=None):
    if not isinstance(anchors_batch, list):
        anchors_batch = len(gt_box_batch) * [anchors_batch]
    if not all(
            isinstance(anchors, box_list.BoxList) for anchors in anchors_batch):
        raise ValueError('anchors_batch must be a BoxList or list of BoxLists.')
    if not (len(anchors_batch)
            == len(gt_box_batch)
            == len(gt_class_targets_batch)):
        raise ValueError('batch size incompatible with lengths of anchors_batch, '
                         'gt_box_batch and gt_class_targets_batch.')
    cls_targets_list = []
    cls_weights_list = []
    reg_targets_list = []
    reg_weights_list = []
    match_list = []
    if gt_weights_batch is None:
        gt_weights_batch = [None] * len(gt_class_targets_batch)
    for anchors, gt_boxes, gt_class_targets, gt_weights in zip(
            anchors_batch, gt_box_batch, gt_class_targets_batch, gt_weights_batch):
        (cls_targets, cls_weights, reg_targets,
         reg_weights, match) = target_assigner.assign(
            anchors, gt_boxes, gt_class_targets, gt_weights)
        cls_targets_list.append(cls_targets)
        cls_weights_list.append(cls_weights)
        reg_targets_list.append(reg_targets)
        reg_weights_list.append(reg_weights)
        match_list.append(match)
    batch_cls_targets = np.stack(cls_targets_list)
    batch_cls_weights = np.stack(cls_weights_list)
    batch_reg_targets = np.stack(reg_targets_list)
    batch_reg_weights = np.stack(reg_weights_list)
    return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
            batch_reg_weights, match_list)
