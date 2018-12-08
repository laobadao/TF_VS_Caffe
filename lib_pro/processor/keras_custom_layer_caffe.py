import tensorflow as tf
from quatization.lib.processor.faster_rcnn_post import KerasCustomLayer
import math

class ProposalCaffe(KerasCustomLayer):

    def __init__(self):
        super(KerasCustomLayer, self).__init__()
        self._feat_stride = 16
        self._anchor_scales = (8, 16, 32)
        self._pre_nms_topN = 6000
        self._post_nms_topN = 300
        self._nms_thresh = 0.7
        self._min_size = 16

    @property
    def class_name(self):
        return 'Proposal'

    def get_config(self):
        config = {
            'feat_stride': self._feat_stride,
            'scales': self._anchor_scales,
            'pre_nms_topN': self._pre_nms_topN,
            'post_nms_topN': self._post_nms_topN,
            'nms_thresh': self._nms_thresh,
            'min_size': self._min_size
        }

        config_data = super(ProposalCaffe, self).get_config()
        config_data.setdefault('config', config)
        return config_data

    def set_config(self, feat_stride=16.0, scales=(8, 16, 32), pre_nms_topN=6000,
                   post_nms_topN=300, nms_thresh=0.5, min_size=16.0):
        self._feat_stride = feat_stride
        self._anchor_scales = scales
        self._pre_nms_topN = pre_nms_topN
        self._post_nms_topN = post_nms_topN
        self._nms_thresh = nms_thresh
        self._min_size = min_size

    def proposal(self, itfuts):
        assert len(itfuts) == 2
        pre_nms_topN = self._pre_nms_topN
        post_nms_topN = self._post_nms_topN
        nms_thresh = self._nms_thresh
        min_size = self._min_size

        _anchors = self._generate_anchors()

        _num_anchors = _anchors.shape[0]
        scores = itfuts[0][:, :, :, _num_anchors:]
        bbox_deltas = itfuts[1]

        # anchors
        height, width = scores.shape[-3:-1]
        shift_x = tf.range(0, width) * self._feat_stride
        shift_y = tf.range(0, height) * self._feat_stride
        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        flatten_shift_x = tf.reshape(shift_x, shape=[-1])
        flatten_shift_y = tf.reshape(shift_y, shape=[-1])
        stacked_shifts = tf.stack((flatten_shift_x, flatten_shift_y,
                                   flatten_shift_x, flatten_shift_y))

        shifts = tf.transpose(stacked_shifts)
        A = _num_anchors
        K = shifts.shape[0]
        shifts = tf.cast(shifts, dtype=tf.float32)
        anchors = tf.reshape(_anchors, shape=(1, A, 4)) + tf.transpose(tf.reshape(shifts, shape=(1, K, 4)), perm=(1, 0, 2))
        anchors = tf.reshape(anchors, (K * A, 4))
        # box
        bbox_deltas = tf.reshape(bbox_deltas, shape=(-1, 4))

        # scores
        scores = tf.reshape(scores, shape=(-1, 1))
        proposals = self._bbox_transform_inv(anchors, bbox_deltas)

        _feat_stride = tf.constant(self._feat_stride, dtype=tf.float32)
        print("height:", height)
        print("width:", width)
        height = tf.cast(height, dtype=tf.float32)
        width = tf.cast(width, dtype=tf.float32)

        # im_info = tf.constant([height * _feat_stride, width * _feat_stride, 1], dtype=tf.float32)
        # keras need
        im_info = tf.stack([height * _feat_stride, width * _feat_stride, 1], axis=0)
        proposals = self._clip_boxes(proposals, im_info[:2])
        keep = self._filter_boxes(proposals, min_size * im_info[2])

        keep = tf.transpose(keep)
        proposals = tf.gather(proposals, indices=keep)[0]
        scores = tf.gather(scores, indices=keep)[0]
        scores = tf.reshape(scores, shape=[-1])
        num_boxes = anchors.shape[0]
        _, order = tf.nn.top_k(scores, k=num_boxes, sorted=True)

        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]

        order = tf.expand_dims(order, axis=0)
        proposals = tf.gather(proposals, indices=order)[0]
        scores = tf.gather(scores, indices=order)[0]
        scores = tf.expand_dims(scores, axis=1)
        stacked_proposal_score = tf.concat((proposals, scores), axis=1)

        keep = self._nms(stacked_proposal_score, nms_thresh, post_nms_topN)
        proposals = tf.gather(proposals, indices=keep)
        # !!! keras need add one dimension
        proposals = tf.expand_dims(proposals, axis=0)
        return proposals

    def _generate_anchors(self, base_size=16, ratios=(0.5, 1, 2),
                          scales=2 ** tf.range(3, 6)):
        """
            Generate anchor (reference) windows by enumerating aspect ratios X
            scales wrt a reference (0, 0, 15, 15) window.
            """
        scales = tf.cast(scales, dtype=tf.float32)
        base_anchor = tf.constant([1, 1, base_size, base_size], dtype=tf.float32) - 1
        ratio_anchors = self._ratio_enum(base_anchor, ratios)
        iter_i = range(0, ratio_anchors.shape[0])
        anchors = tf.concat([self._scale_enum(ratio_anchors[i, :], scales) for i in iter_i], axis=0)

        return anchors

    def _whctrs(self, anchor):
        """
        Return width, height, x center, and y center for an anchor (window).
        """

        w = anchor[2] - anchor[0] + 1
        h = anchor[3] - anchor[1] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        y_ctr = anchor[1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def _mkanchors(self, ws, hs, x_ctr, y_ctr):
        """
        Given a vector of widths (ws) and heights (hs) around a center
        (x_ctr, y_ctr), output a set of anchors (windows).
        """
        anchors = tf.stack((x_ctr - 0.5 * (ws - 1),
                            y_ctr - 0.5 * (hs - 1),
                            x_ctr + 0.5 * (ws - 1),
                            y_ctr + 0.5 * (hs - 1)), axis=1)

        return anchors

    def _ratio_enum(self, anchor, ratios):
        """
        Enumerate a set of anchors for each aspect ratio wrt an anchor.
        """

        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        size = w * h
        size_ratios = size / ratios
        sqrted_size_ratios = tf.sqrt(size_ratios)
        ws = tf.round(sqrted_size_ratios)
        hs = tf.round(ws * ratios)

        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _scale_enum(self, anchor, scales):
        """
        Enumerate a set of anchors for each scale wrt an anchor.
        """
        w, h, x_ctr, y_ctr = self._whctrs(anchor)
        ws = w * scales
        hs = h * scales
        anchors = self._mkanchors(ws, hs, x_ctr, y_ctr)
        return anchors

    def _bbox_transform_inv(self, boxes, deltas):
        if boxes.shape[0] == 0:
            return tf.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        boxes = tf.cast(boxes, dtype=deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * widths[:, tf.newaxis] + ctr_x[:, tf.newaxis]
        pred_ctr_y = dy * heights[:, tf.newaxis] + ctr_y[:, tf.newaxis]
        pred_w = tf.exp(dw) * widths[:, tf.newaxis]
        pred_h = tf.exp(dh) * heights[:, tf.newaxis]

        # x1
        pred_boxes_0 = pred_ctr_x - 0.5 * pred_w
        print("pred_boxes_0.shape: ", pred_boxes_0.shape)
        # y1
        pred_boxes_1 = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes_2 = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes_3 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = tf.concat([pred_boxes_0, pred_boxes_1, pred_boxes_2, pred_boxes_3], axis=1)

        print("===== bbox_transform_inv pred_boxes:", pred_boxes.shape)

        return pred_boxes

    def _clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """

        # x1 >= 0
        boxes_0 = tf.maximum(tf.minimum(boxes[:, 0::4], im_shape[1] - 1), 0.0)
        # y1 >= 0
        boxes_1 = tf.maximum(tf.minimum(boxes[:, 1::4], im_shape[0] - 1), 0.0)
        # x2 < im_shape[1]
        boxes_2 = tf.maximum(tf.minimum(boxes[:, 2::4], im_shape[1] - 1), 0.0)
        # y2 < im_shape[0]
        boxes_3 = tf.maximum(tf.minimum(boxes[:, 3::4], im_shape[0] - 1), 0.0)
        boxes = tf.concat([boxes_0, boxes_1, boxes_2, boxes_3], axis=1)
        return boxes

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = tf.where((ws >= min_size) & (hs >= min_size))
        return keep

    def _nms(self, dets, thresh, max_output_size):
        """Pure Python NMS baseline."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        num = scores.shape[0]
        _, order = tf.nn.top_k(scores, k=num, sorted=True)

        keep = []
        while max_output_size > 0:
            i = order[0]
            keep.append(i)

            xx1 = tf.maximum(x1[i], tf.gather(x1, indices=order[1:]))
            yy1 = tf.maximum(y1[i], tf.gather(y1, indices=order[1:]))
            xx2 = tf.minimum(x2[i], tf.gather(x2, indices=order[1:]))
            yy2 = tf.minimum(y2[i], tf.gather(y2, indices=order[1:]))

            w = tf.maximum(0.0, xx2 - xx1 + 1)
            h = tf.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + tf.gather(areas, indices=order[1:]) - inter)
            inds = tf.reshape(tf.where(ovr <= thresh), shape=[-1])
            inds = inds + 1
            order = tf.gather(order, indices=inds)
            max_output_size -= 1

        return keep


class RoiPoolingCaffe(KerasCustomLayer):
    def __init__(self):
        super(KerasCustomLayer, self).__init__()
        self._pooled_w = 7
        self._pooled_h = 7
        self._spatial_scale = 0.0625

    @property
    def class_name(self):
        return 'ROIPooling'

    def get_config(self):
        config = {
            'pooled_w': self._pooled_w,
            'pooled_h': self._pooled_h,
            'spatial_scale': self._spatial_scale,
        }

        config_data = super(RoiPoolingCaffe, self).get_config()
        config_data.setdefault('config', config)
        return config_data

    def set_config(self, pooled_w=7, pooled_h=7, spatial_scale=0.0625):
        self._pooled_w = pooled_w
        self._pooled_h = pooled_h
        self._spatial_scale = spatial_scale

    def roi_pooling(self, inputs):

        assert (len(inputs) == 2)

        img = inputs[0]
        rois = inputs[1]

        print('------------------img shape = ', img.shape)
        print("-----------------rois = ", rois.shape)
        num_rois = rois.shape[1]
        print("num_rois:", num_rois)
        im_h, im_w = img.shape[1:3]
        im_h = tf.constant(im_h.value, dtype=tf.float32)
        im_w = tf.constant(im_w.value, dtype=tf.float32)

        i_r_list = []
        # num_rois
        for i_r in range(num_rois):

            roi_x1 = tf.round(rois[0, i_r, 0] * self._spatial_scale)
            roi_y1 = tf.round(rois[0, i_r, 1] * self._spatial_scale)
            roi_x2 = tf.round(rois[0, i_r, 2] * self._spatial_scale)
            roi_y2 = tf.round(rois[0, i_r, 3] * self._spatial_scale)



            roi_h = tf.maximum(roi_y2 - roi_y1 + 1, 1)
            roi_w = tf.maximum(roi_x2 - roi_x1 + 1, 1)

            bin_size_h = roi_h / self._pooled_h
            bin_size_w = roi_w / self._pooled_w

            ph_list = []
            for ph in range(self._pooled_h):
                pw_list = []
                for pw in range(self._pooled_w):
                    x1 = tf.cast(tf.floor(tf.minimum(tf.maximum(pw * bin_size_w + roi_x1, 0), im_w)), dtype=tf.int32)
                    y1 = tf.cast(tf.floor(tf.minimum(tf.maximum(ph * bin_size_h + roi_y1, 0), im_h)), dtype=tf.int32)
                    x2 = tf.cast(tf.ceil(tf.minimum(tf.maximum((pw + 1) * bin_size_w + roi_x1, 0), im_w)), dtype=tf.int32)
                    y2 = tf.cast(tf.ceil(tf.minimum(tf.maximum((ph + 1) * bin_size_h + roi_y1, 0), im_h)), dtype=tf.int32)

                    crop = tf.slice(img, begin=[0, y1, x1, 0], size=[-1, y2-y1, x2-x1, -1])
                    pooled_val = tf.reduce_max(input_tensor=crop, reduction_indices=[1, 2])
                    pw_list.append(pooled_val)

                outputs_pw = tf.stack(pw_list, axis=1)
                ph_list.append(outputs_pw)
            outputs_ph = tf.stack(ph_list, axis=1)
            i_r_list.append(outputs_ph)
        outputs = tf.stack(i_r_list, axis=1)
        return outputs

