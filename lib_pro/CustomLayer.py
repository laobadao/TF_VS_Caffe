# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from keras.engine.topology import Layer
import tensorflow as tf


class ProposalTF(Layer):
    def __init__(self, scales=(0.25, 0.5, 1.0, 2.0), aspect_ratios=(0.5, 1.0, 2.0), anchor_stride=(16, 16),
                 max_proposals=100, nms_score_threshold=0.0, nms_iou_threshold=0.699999988079, **kwargs):
        super(ProposalTF, self).__init__(**kwargs)
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.anchor_stride = anchor_stride
        self.max_proposals = max_proposals
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def compute_output_shape(self, input_shape):
        return [None, self.max_proposals, 4]

    def call(self, inputs, **kwargs):
        from quatization.lib.processor.faster_rcnn_post import ProposalTF
        assert len(inputs) == 5

        proposal_tf = ProposalTF()
        proposal_tf.set_config(self.scales, self.aspect_ratios, self.anchor_stride, self.max_proposals,
                               self.nms_score_threshold, self.nms_iou_threshold)

        proposal_boxes_normalized = proposal_tf.proposal(preprocessed_inputs=inputs[0],
                                                         box_encodings=inputs[1],
                                                         class_predictions_with_background=inputs[2],
                                                         rpn_box_predictor_features=inputs[3],
                                                         rpn_features_to_crop=inputs[4])
        return proposal_boxes_normalized

    def get_config(self):
        config = {'scales': self.scales,
                  'aspect_ratios': self.aspect_ratios,
                  'anchor_stride': self.anchor_stride,
                  'max_proposals': self.max_proposals,
                  'nms_score_threshold': self.nms_score_threshold,
                  'nms_iou_threshold': self.nms_iou_threshold}
        base_config = super(ProposalTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RoiPoolingTF(Layer):
    def __init__(self, initial_crop_size=14, maxpool_kernel_size=2, maxpool_stride=2, **kwargs):
        super(RoiPoolingTF, self).__init__(**kwargs)
        self.initial_crop_size = initial_crop_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride

    def compute_output_shape(self, input_shape):
        out_h_w = (self.initial_crop_size - self.maxpool_kernel_size) // self.maxpool_stride + 1
        assert len(input_shape[1]) == 3
        c = input_shape[0][-1]
        batch = input_shape[1][1]
        return [None, batch, out_h_w, out_h_w, c]

    def call(self, inputs, **kwargs):
        from quatization.lib.processor.faster_rcnn_post import RoiPoolingTF
        assert len(inputs) == 2

        roi_pooling_tf = RoiPoolingTF()
        roi_pooling_tf.set_config(initial_crop_size=self.initial_crop_size,
                                  maxpool_kernel_size=self.maxpool_kernel_size,
                                  maxpool_stride=self.maxpool_stride)
        return roi_pooling_tf.roi_pooling(features_to_crop=inputs[0], proposal_boxes_normalized=inputs[1])

    def get_config(self):
        config = {'initial_crop_size': self.initial_crop_size,
                  'maxpool_kernel_size': self.maxpool_kernel_size,
                  'maxpool_stride': self.maxpool_stride}
        base_config = super(RoiPoolingTF, self).get_config()
        return dict(list(base_config.items() + list(config.items())))


class ProposalCaffe(Layer):
    def __init__(self, feat_stride=16, scales=(8, 16, 32), pre_nms_topN=6000,
                 post_nms_topN=300, nms_thresh=0.5, min_size=16, **kwargs):
        super(ProposalCaffe, self).__init__(**kwargs)
        self._feat_stride = feat_stride
        self._anchor_scales = scales
        self._pre_nms_topN = pre_nms_topN
        self._post_nms_topN = post_nms_topN
        self._nms_thresh = nms_thresh
        self._min_size = min_size

    def compute_output_shape(self, input_shape):
        return [None, self._post_nms_topN, 4]

    def call(self, inputs, **kwargs):
        from quatization.lib.processor.keras_custom_layer_caffe import ProposalCaffe
        assert len(inputs) == 2

        proposal_caffe = ProposalCaffe()
        proposal_caffe.set_config(feat_stride=self._feat_stride, scales=self._anchor_scales,
                                  pre_nms_topN=self._pre_nms_topN, post_nms_topN=self._post_nms_topN,
                                  nms_thresh=self._nms_thresh, min_size=self._min_size)
        proposal_boxes_normalized = proposal_caffe.proposal(inputs)
        return proposal_boxes_normalized

    def get_config(self):
        config = {
            'feat_stride': self._feat_stride,
            'scales': self._anchor_scales,
            'pre_nms_topN': self._pre_nms_topN,
            'post_nms_topN': self._post_nms_topN,
            'nms_thresh': self._nms_thresh,
            'min_size': self._min_size
        }
        base_config = super(ProposalCaffe, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RoiPoolingCaffe(Layer):
    def __init__(self, pooled_w=7, pooled_h=7, spatial_scale=0.0625, **kwargs):
        super(RoiPoolingCaffe, self).__init__(**kwargs)
        self._pooled_w = pooled_w
        self._pooled_h = pooled_h
        self._spatial_scale = spatial_scale

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        c = input_shape[1][-1]
        return [None, batch, self._pooled_h, self._pooled_w, c]

    def call(self, inputs, **kwargs):
        from quatization.lib.processor.keras_custom_layer_caffe import RoiPoolingCaffe
        assert len(inputs) == 2

        roi_pooling_caffe = RoiPoolingCaffe()
        roi_pooling_caffe.set_config(pooled_w=self._pooled_w, pooled_h=self._pooled_h,
                                     spatial_scale=self._spatial_scale)
        return roi_pooling_caffe.roi_pooling(inputs)

    def get_config(self):
        config = {
            'pooled_w': self._pooled_w,
            'pooled_h': self._pooled_h,
            'spatial_scale': self._spatial_scale,
        }
        base_config = super(RoiPoolingCaffe, self).get_config()
        return dict(list(base_config.items() + list(config.items())))


class SpaceToDepth(Layer):
    def __init__(self, block_size, **kwargs):
        super(SpaceToDepth, self).__init__(**kwargs)
        self.block_size = int(block_size)

    def compute_output_shape(self, input_shape):
        block_size = self.block_size
        return (
            None, input_shape[1] / block_size, input_shape[2] / block_size, input_shape[3] * block_size * block_size)

    def call(self, inputs, **kwargs):
        return tf.space_to_depth(inputs, block_size=self.block_size)

    def get_config(self):
        config = {'block_size': self.block_size}
        base_config = super(SpaceToDepth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Squeeze(Layer):
    def __init__(self, axis, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape):
        out_shape = []
        for i in range(len(input_shape)):
            if i not in self.axis:
                out_shape.append(input_shape[i])
        return out_shape

    def call(self, inputs, **kwargs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Squeeze, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Mul(Layer):
    def __init__(self, scale, **kwargs):
        super(Mul, self).__init__(**kwargs)
        self.scale = scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return self.scale * inputs

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(Mul, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
