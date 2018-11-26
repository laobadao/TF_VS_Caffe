# -*- coding: utf-8 -*-

import numpy as np
import math

import config
from rpn.generate_anchors_tf import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv_tf, clip_to_window, clip_boxes
from fast_rcnn.nms_wrapper import nms


class TensorflowCustomLayer(object):
    """docstring for MiddleLayer"""

    def __init__(self, name):
        self._name = name
        self._inputs = []
        self._output = None

    @property
    def sub_type(self):
        return 'custom_layer'

    @property
    def class_name(self):
        return 'Custom'

    @property
    def inputs(self):
        return self._inputs

    @property
    def output(self):
        return self._output

    def translate(self, caffe_net, pretrained_net, first_layer, last_layer, input_dims, input_cache):
        pass

    def get_config(self):
        config_data = {
            'class_name': self.class_name,
            'inputs': self._inputs,
            'output': self._output,
        }
        return config_data

    def _caffe_layer(self, caffe_net, first_layer, last_layer, input_cache):
        for layer in caffe_net.layer:
            if layer.name == first_layer:
                self._set_io(layer, input_cache)  # could be better
                return layer
        return None

    def _set_io(self, layer, input_cache):
        for b in layer.bottom:
            if input_cache.has_key(b):
                b = input_cache[b]
            self._inputs.append(b)

        self._output = layer.top[0]


class TensorflowProposal(TensorflowCustomLayer):
    """docstring for Proposal"""

    def __init__(self, name):
        super(TensorflowProposal, self).__init__(name)

    @property
    def class_name(self):
        return 'Proposal'

    def get_config(self):
        config = {
            'feat_stride': self._feat_stride,
            'scales': self._anchor_scales
        }

        config_data = super(TensorflowProposal, self).get_config()
        config_data.setdefault('config', config)
        return config_data

    def from_config(self, feat_stride=16, scales=(4, 8, 16, 32)):
        self._feat_stride = feat_stride
        self._anchor_scales = scales

    def predict(self, inputs):
        # class_prediction, inputs[0] (1, 38, 63, 24))
        # box_encodings, inputs[1] (1, 38, 63, 48))
        # image_shape
        print('input_0.shape=', inputs[0].shape)
        print('input_1.shape=', inputs[1].shape)
        print('input_2.shape=', inputs[2])
        image_shape = inputs[2]
        scales = config.cfg.POSTPROCESSOR.SCALES
        aspect_ratios = config.cfg.POSTPROCESSOR.ASPECT_RATIOS
        height_stride = config.cfg.POSTPROCESSOR.HEIGHT_STRIDE
        width_stride = config.cfg.POSTPROCESSOR.WIDTH_STRIDE

        _num_anchors = len(scales) * len(aspect_ratios)
        print("_num_anchors:", _num_anchors)
        scores = inputs[0][:, :, :, _num_anchors:]
        bbox_deltas = inputs[1]
        # box
        bbox_deltas = bbox_deltas.reshape((-1, 4))

        # scores
        scores = scores.reshape((-1, 1))
        print("scores:", scores.shape)
        # anchors
        height, width = inputs[0].shape[1], inputs[0].shape[2]

        feature_map_shape_list = [(height, width)]

        anchors = generate_anchors(scales=[scale for scale in scales],
                                   aspect_ratios=[aspect_ratio
                                                  for aspect_ratio
                                                  in aspect_ratios],
                                   base_anchor_size=None,
                                   anchor_stride=[height_stride,
                                                   width_stride],
                                   anchor_offset=None, feature_map_shape_list=feature_map_shape_list)

        pre_nms_topN = 6000
        post_nms_topN = 100
        nms_thresh = 0.699999988079
        min_size = 16
        #  box_encodings, inputs[1] (1, 38, 63, 48))
        # bbox_deltas:', (28728, 4)
        # clip_window:', array([   0,    0,  600, 1002]))

        # tf clip_to_window
        print("============== clip_to_window ===================")

        proposals = bbox_transform_inv_tf(anchors, bbox_deltas)

        boxdecode = proposals

        clip_window = np.array([0, 0, height, width])
        print("clip_window:", clip_window)
        # ('proposals_clip  :', (1829, 4))
        proposals_clip = clip_to_window(proposals, clip_window)

        print("proposals_clip clip_to_window :", proposals_clip.shape)
        print("proposals_clip clip_to_window[0] :", proposals_clip[0])

        # anchors[0]:', array([ 0.      ,  0.      , 45.254834, 22.627417]))

        # proposals1 0:', array([11.60263 ,  3.129001, 41.311607, 18.966888], dtype=float32))
        # ('proposals1 1:', array([  0.22217222,   1.6537127 , 100.95798   ,  44.21667   ],
        #       dtype=float32))

        boxdecode = proposals

        im_info = np.array([height, width, 0])
        # im_info:', array([38, 63,  0]))
        proposals = clip_boxes(proposals, im_info[:2])

        print("proposals clip_boxes :", proposals.shape)
        print("proposals clip_boxes [0]:", proposals[0])

        # im_info:[:2]', array([38, 63]

        print("im_info:[:2]", im_info[:2])
        keep = self._filter_boxes(proposals, min_size * im_info[2])

        proposals = proposals[keep, :]
        print("proposals3:", proposals.shape)

        # 'scores.shape1', (28728, 1

        print("scores.shape1", scores.shape)
        scores = scores[keep]
        print("scores.shape2", scores.shape)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]

        # proposals4:', (6000, 4))
        print("proposals4 pre:", proposals.shape)
        scores = scores[order]
        # TODO nms 方法是否重写
        keep = nms(np.hstack((proposals, scores)), nms_thresh)

        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        # ('proposals5:', (100, 4))
        print("proposals final:", proposals.shape)
        return proposals, boxdecode, anchors

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        print("boxes:", boxes.shape)
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

class TensorflowROIPooling(TensorflowCustomLayer):
    """docstring for ROIPooling"""

    def __init__(self, name):
        super(TensorflowROIPooling, self).__init__(name)

    @property
    def class_name(self):
        return 'ROIPooling'

    def from_config(self, pooled_w=7, pooled_h=7, spatial_scale=0.0625):
        self._pooled_w = pooled_w
        self._pooled_h = pooled_h
        self._spatial_scale = spatial_scale

    def get_config(self):
        config = {
            'pooled_w': self._pooled_w,
            'pooled_h': self._pooled_h,
            'spatial_scale': self._spatial_scale,
        }

        config_data = super(TensorflowROIPooling, self).get_config()
        config_data.setdefault('config', config)
        return config_data

    def predict(self, inputs):
        assert (len(inputs) == 2)

        img = inputs[0]
        rois = inputs[1]

        print('img shape = ', img.shape)
        num_rois = len(rois)
        im_h, im_w = img.shape[1], img.shape[2]

        outputs = np.zeros((num_rois, self._pooled_h, self._pooled_w, img.shape[3]))

        for i_r in range(num_rois):
            roi_x1 = round(rois[i_r, 0] * self._spatial_scale)
            roi_y1 = round(rois[i_r, 1] * self._spatial_scale)
            roi_x2 = round(rois[i_r, 2] * self._spatial_scale)
            roi_y2 = round(rois[i_r, 3] * self._spatial_scale)

            roi_h = max(roi_y2 - roi_y1 + 1, 1)
            roi_w = max(roi_x2 - roi_x1 + 1, 1)

            bin_size_h = roi_h / float(self._pooled_h)
            bin_size_w = roi_w / float(self._pooled_w)

            for ph in range(self._pooled_h):
                for pw in range(self._pooled_w):
                    x1 = int(math.floor(min(max(pw * bin_size_w + roi_x1, 0), im_w)))
                    y1 = int(math.floor(min(max(ph * bin_size_h + roi_y1, 0), im_h)))
                    x2 = int(math.ceil(min(max((pw + 1) * bin_size_w + roi_x1, 0), im_w)))
                    y2 = int(math.ceil(min(max((ph + 1) * bin_size_h + roi_y1, 0), im_h)))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = img[0, y1:y2, x1:x2, :]
                    pooled_val = np.max(np.max(crop, axis=0), axis=0)
                    outputs[i_r, ph, pw, :] = pooled_val
        return outputs
