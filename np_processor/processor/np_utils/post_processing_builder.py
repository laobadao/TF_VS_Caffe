"""Builder function for post processing operations."""
import functools
import numpy as np
from ..np_utils import post_processing, ops
from platformx.plat_tensorflow.tools.processor import model_config
import config

def build(model_type):

    if model_type == model_config.SSD:
        score_converter = "SIGMOID"
    elif model_type == model_config.FASTER_RCNN:
        score_converter = "SOFTMAX"
    else:
        raise ValueError('type must be ssd or faster_rcnn string')

    non_max_suppressor_fn = _build_non_max_suppressor(model_type)
    # optional float logit_scale = 3[default = 1.0];
    logit_scale = 1.0
    score_converter_fn = _build_score_converter(score_converter, logit_scale)
    return non_max_suppressor_fn, score_converter_fn


def _build_non_max_suppressor(type):
    """Builds non-max suppresson based on the nms config.

    Args:
      nms_config: post_processing_pb2.PostProcessing.BatchNonMaxSuppression proto.

    Returns:
      non_max_suppressor_fn: Callable non-max suppressor.

    Raises:
      ValueError: On incorrect iou_threshold or on incompatible values of
        max_total_detections and max_detections_per_class.
    """

    if type == model_config.SSD:
        score_threshold = config.cfg.POSTPROCESSOR.SCORE_THRESHOLD
        iou_threshold = config.cfg.POSTPROCESSOR.IOU_THRESHOLD
        max_detections_per_class = config.cfg.POSTPROCESSOR.MAX_DETECTIONS_PER_CLASS
        max_total_detections = config.cfg.POSTPROCESSOR.MAX_TOTAL_DETECTIONS
    elif type == model_config.FASTER_RCNN:
        score_threshold = config.cfg.POSTPROCESSOR.SCORE_THRESHOLD
        iou_threshold = config.cfg.POSTPROCESSOR.IOU_THRESHOLD
        max_detections_per_class = config.cfg.POSTPROCESSOR.MAX_DETECTIONS_PER_CLASS
        max_total_detections = config.cfg.POSTPROCESSOR.MAX_TOTAL_DETECTIONS
    else:
        raise ValueError('type must be ssd or faster_rcnn string')

    if iou_threshold < 0 or iou_threshold > 1.0:
        raise ValueError('iou_threshold not in [0, 1.0].')
    if max_detections_per_class > max_total_detections:
        raise ValueError('max_detections_per_class should be no greater than '
                         'max_total_detections.')

    non_max_suppressor_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=score_threshold,
        iou_thresh=iou_threshold,
        max_size_per_class=max_detections_per_class,
        max_total_size=max_total_detections)

    return non_max_suppressor_fn


def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
    """Create a function to scale logits then apply a Tensorflow function."""

    def score_converter_fn(logits):
        scaled_logits = np.divide(logits, logit_scale)

        return tf_score_converter_fn(scaled_logits)

    return score_converter_fn


def _build_score_converter(score_converter, logit_scale):
    """Builds score converter based on the config.

    Builds one of [tf.identity, tf.sigmoid, tf.softmax] score converters based on
    the config.

    Args:
      score_converter_config: post_processing_pb2.PostProcessing.score_converter.
      logit_scale: temperature to use for SOFTMAX score_converter.

    Returns:
      Callable score converter op.

    Raises:
      ValueError: On unknown score converter.
    """


    if score_converter == "SIGMOID":
        return _score_converter_fn_with_logit_scale(ops.sigmoid, logit_scale)
    elif score_converter == "SOFTMAX":
        return _score_converter_fn_with_logit_scale(ops.softmax, logit_scale)

    raise ValueError('Unknown score converter.')

