import numpy as np
import math

from config import cfg
from rpn.generate_anchors import generate_anchors
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
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

        _anchors = generate_anchors(scales=np.array(self._anchor_scales))
        _num_anchors = _anchors.shape[0]

        print("_num_anchors", _anchors.shape)

        pre_nms_topN = 6000
        post_nms_topN = 100
        nms_thresh = 0.699999988079
        min_size = 16

        print("nms_thresh", nms_thresh)
        scores = inputs[0][:, :, :, _num_anchors:]
        bbox_deltas = inputs[1]

        # anchors
        height, width = scores.shape[-3:-1]
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        A = _num_anchors
        K = shifts.shape[0]
        anchors = _anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        print("================ anchors", anchors.shape)
        print("================ anchors 0", anchors[0])
        # box
        bbox_deltas = bbox_deltas.reshape((-1, 4))

        # scores
        scores = scores.reshape((-1, 1))

        proposals = bbox_transform_inv(anchors, bbox_deltas)
        boxdecode = proposals

        im_info = np.array([height, width, 0])
        print("bbox_transform_inv proposals:", proposals.shape)
        print("bbox_transform_inv proposals[0]:", proposals[0])

        proposals = clip_boxes(proposals, im_info[:2])
        print("clip_boxes proposals :", proposals.shape)
        print("clip_boxes proposals[0] :", proposals[0])
        print("im_info:[:2]", im_info[:2])
        keep = self._filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        print("_filter_boxes proposals3:", proposals.shape)
        print("_filter_boxes proposals3 [0]:", proposals[0])

        print("scores.shape1", scores.shape)
        scores = scores[keep]
        print("scores.shape2", scores.shape)
        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        print("proposals4:", proposals.shape)
        scores = scores[order]

        keep = nms(np.hstack((proposals, scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]

        print("proposals5:", proposals.shape)
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
