# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
import numpy as np
from platformx.plat_tensorflow.tools.processor.np_utils import shape_utils, \
    anchor_generator_builder, box_list_ops, box_list, ops, post_processing_builder, \
    target_assigner, post_processing, visualization_utils as vis_util
from platformx.plat_tensorflow.tools.processor.np_utils import standard_fields as fields
from platformx.plat_tensorflow.tools.processor import model_config
import config
from PIL import Image
from platformx.plat_tensorflow.tools.processor.np_utils import label_map_util
from scipy import misc
import os
import matplotlib
matplotlib.use('Agg')


BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'

PATH_TO_LABELS = config.cfg.POSTPROCESSOR.PATH_TO_LABELS

BOXES_NAME = "FirstStageBoxPredictor_BoxEncodingPredictor"
CLASSES_NAME = "FirstStageBoxPredictor_ClassPredictor"
FEATURES_NAME = "FirstStageFeatureExtractor"
BOX_PREDICTOR = "SecondStageBoxPredictor_Reshape_1"


def faster_rcnn_stage_one_post(preprocessed_inputs, result_middle=None):
    print("========================== faster_rcnn_stage_one_post ========================== ")
    preprocessed_inputs = preprocessed_inputs
    print("1 preprocessed_inputs:", preprocessed_inputs.shape)

    for key, value in result_middle.items():
        if BOXES_NAME in key:
            box_encodings = value
            print("box_encodings:", value.shape)

        if CLASSES_NAME in key:
            class_predictions_with_background = value
            print("class_predictions_with_background:", value.shape)

        if FEATURES_NAME in key:
            rpn_features_to_crop = value
            print("rpn_features_to_crop:", value.shape)

        if BOXES_NAME not in key and CLASSES_NAME not in key and FEATURES_NAME not in key:
            rpn_box_predictor_features = value
            print("rpn_box_predictor_features:", value.shape)

    fisrt_post_result = crop_and_resize_to_input(rpn_box_predictor_features, preprocessed_inputs, box_encodings,
                                                 class_predictions_with_background, rpn_features_to_crop)

    return fisrt_post_result


def faster_rcnn_second_stage_post(preprocessed_inputs, result_middle=None, second_net_result=None):
    print("================= faster_rcnn_second_stage_post =================")
    preprocessed_inputs = preprocessed_inputs
    print("preprocessed_inputs.shape:", preprocessed_inputs.shape)

    for key, value in result_middle.items():
        if BOXES_NAME in key:
            box_encodings = value
            print("box_encodings:", value.shape)
        if CLASSES_NAME in key:
            class_predictions_with_background = value
            print("class_predictions_with_background:", value.shape)
        if FEATURES_NAME in key:
            rpn_features_to_crop = value
            print("rpn_features_to_crop:", value.shape)
        if BOXES_NAME not in key and CLASSES_NAME not in key and FEATURES_NAME not in key:
            rpn_box_predictor_features = value
            print("rpn_box_predictor_features:", value.shape)

    for key, value in second_net_result.items():
        if BOX_PREDICTOR in key:
            if value.ndim == 4:
                value = np.transpose(value, axes=(0, 2, 3, 1))

            if value.ndim == 3:
                value = np.expand_dims(value, axis=3)
            class_prediction_reshape = value
            print("class_prediction_reshape.shape:", value.shape)
        if BOX_PREDICTOR not in key:

            print(" before box_encoding_reshape.shape:", value.shape)

            if value.ndim == 4 and value.shape[3] != 4:
                value = value.reshape((value.shape[0], value.shape[2], value.shape[3], value.shape[1]))
            box_encoding_reshape = value

            print("box_encoding_reshape.shape:", box_encoding_reshape.shape)

    input_shape = preprocessed_inputs.shape
    true_image_shapes = np.array([[input_shape[1], input_shape[2], input_shape[3]]], dtype=np.int32)
    print("2 true_image_shapes:", true_image_shapes)

    result_output = second_stage_box_predictor(preprocessed_inputs, box_encoding_reshape, class_prediction_reshape,
                                               rpn_features_to_crop, box_encodings, class_predictions_with_background,
                                               true_image_shapes, rpn_box_predictor_features)
    result_show = result_output
    show_detection_result(result_show)

    return result_output


def second_stage_box_predictor(preprocessed_inputs, box_encoding_reshape, class_prediction_reshape,
                               rpn_features_to_crop,
                               rpn_box_encodings,
                               rpn_objectness_predictions_with_background,
                               true_image_shapes,
                               rpn_box_predictor_features):
    image_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_inputs)

    first_stage_anchor_generator = anchor_generator_builder.build()

    clip_window = np.stack([0, 0, image_shape[1], image_shape[2]])
    feature_map_shape = rpn_features_to_crop.shape

    anchors_boxlist = box_list_ops.concatenate(
        first_stage_anchor_generator.generate([(feature_map_shape[1],
                                                feature_map_shape[2])]))
    anchors_boxlist = box_list_ops.clip_to_window(
        anchors_boxlist, clip_window)
    _anchors = anchors_boxlist

    image_shape_2d = _image_batch_shape_2d(image_shape)

    num_anchors_per_location = (
        first_stage_anchor_generator.num_anchors_per_location())

    if len(num_anchors_per_location) != 1:
        raise RuntimeError('anchor_generator is expected to generate anchors '
                           'corresponding to a single feature map.')
    box_predictions = _first_stage_box_predictor_predict([rpn_box_predictor_features], [rpn_box_encodings],
                                                         [rpn_objectness_predictions_with_background],
                                                         num_anchors_per_location)

    predictions_box_encodings = np.concatenate(
        box_predictions[BOX_ENCODINGS], axis=1)

    rpn_box_encodings = np.squeeze(predictions_box_encodings, axis=2)

    rpn_objectness_predictions_with_background = np.concatenate(
        box_predictions[CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)

    first_stage_max_proposals = config.cfg.POSTPROCESSOR.FIRST_STAGE_MAX_PROPOSALS

    proposal_boxes_normalized, _, num_proposals = _postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        _anchors.get(), image_shape_2d, first_stage_max_proposals=first_stage_max_proposals)

    print("proposal_boxes_normalized:", proposal_boxes_normalized.shape)

    prediction_dict = {
        'rpn_box_predictor_features': rpn_box_predictor_features,
        'rpn_features_to_crop': rpn_features_to_crop,
        'image_shape': image_shape,
        'rpn_box_encodings': rpn_box_encodings,
        'rpn_objectness_predictions_with_background':
            rpn_objectness_predictions_with_background,
    }
    print("=========== box_encoding_reshape", box_encoding_reshape.shape)
    refined_box_encodings = np.squeeze(
        box_encoding_reshape,
        axis=1)
    print("=========== class_prediction_reshape", class_prediction_reshape.shape)
    class_predictions_with_background = np.squeeze(
        class_prediction_reshape,
        axis=1)

    _parallel_iterations = 16

    proposal_boxes_normalized = proposal_boxes_normalized[0]

    absolute_proposal_boxes = ops.normalized_to_image_coordinates(
        proposal_boxes_normalized, image_shape, _parallel_iterations)

    prediction_dict1 = {
        'refined_box_encodings': refined_box_encodings,
        'class_predictions_with_background':
            class_predictions_with_background,
        'num_proposals': num_proposals,
        'proposal_boxes': absolute_proposal_boxes,
    }

    prediction_dict.update(prediction_dict1)
    result_output = second_postprocess(prediction_dict, true_image_shapes)

    return result_output


def second_postprocess(prediction_dict, true_image_shapes):
    postprocessed_tensors = _postprocess_box_classifier(
        prediction_dict['refined_box_encodings'],
        prediction_dict['class_predictions_with_background'],
        prediction_dict['proposal_boxes'],
        prediction_dict['num_proposals'],
        true_image_shapes,
        mask_predictions=None)

    return _add_output_tensor_nodes(postprocessed_tensors)


def _postprocess_box_classifier(
        refined_box_encodings,
        class_predictions_with_background,
        proposal_boxes,
        num_proposals,
        image_shapes,
        mask_predictions=None):
    _first_stage_max_proposals = config.cfg.POSTPROCESSOR.FIRST_STAGE_MAX_PROPOSALS

    max_num_proposals = _first_stage_max_proposals

    num_classes = config.cfg.POSTPROCESSOR.NUM_CLASSES

    _proposal_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'proposal')
    _box_coder = _proposal_target_assigner.box_coder

    _second_stage_nms_fn, second_stage_score_conversion_fn = post_processing_builder.build(model_config.FASTER_RCNN)

    refined_box_encodings_batch = np.reshape(
        refined_box_encodings,
        [-1,
         max_num_proposals,
         refined_box_encodings.shape[1],
         _box_coder.code_size])

    class_predictions_with_background_batch = np.reshape(
        class_predictions_with_background,
        [-1, max_num_proposals, num_classes + 1]
    )

    refined_decoded_boxes_batch = _batch_decode_boxes(
        refined_box_encodings_batch, proposal_boxes)

    class_predictions_with_background_batch = (
        second_stage_score_conversion_fn(
            class_predictions_with_background_batch))

    new_shape = [-1, max_num_proposals, num_classes]
    sliced = class_predictions_with_background_batch[0:, 0:, 1:]
    class_predictions_batch = np.reshape(sliced, new_shape)

    clip_window = _compute_clip_window(image_shapes)

    mask_predictions_batch = None
    if mask_predictions is not None:
        mask_height = mask_predictions.shape[2].value
        mask_width = mask_predictions.shape[3].value

        mask_predictions = ops.sigmoid(mask_predictions)

        mask_predictions_batch = np.reshape(
            mask_predictions, [-1, max_num_proposals,
                               num_classes, mask_height, mask_width])

    (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, _,
     num_detections) = _second_stage_nms_fn(
        refined_decoded_boxes_batch,
        class_predictions_batch,
        clip_window=clip_window,
        change_coordinate_frame=True,
        num_valid_boxes=num_proposals,
        masks=mask_predictions_batch)

    print("========== nmsed_boxes:", nmsed_boxes.shape)
    print("========== nmsed_scores:", nmsed_scores.shape)
    print("========== nmsed_classes:", nmsed_classes.shape)
    print("========== num_detections:", num_detections)

    detections = {
        fields.DetectionResultFields.detection_boxes: nmsed_boxes,
        fields.DetectionResultFields.detection_scores: nmsed_scores,
        fields.DetectionResultFields.detection_classes: nmsed_classes,
        fields.DetectionResultFields.num_detections: num_detections
    }
    if nmsed_masks is not None:
        detections[fields.DetectionResultFields.detection_masks] = nmsed_masks
    return detections


def crop_and_resize_to_input(rpn_box_predictor_features, preprocessed_inputs, box_encodings,
                             class_predictions_with_background, rpn_features_to_crop):
    image_shape = preprocessed_inputs.shape

    first_stage_anchor_generator = anchor_generator_builder.build()

    num_anchors_per_location = (
        first_stage_anchor_generator.num_anchors_per_location())

    # 12 num anchors
    print("num_anchors_per_location:", num_anchors_per_location)

    if len(num_anchors_per_location) != 1:
        raise RuntimeError('anchor_generator is expected to generate anchors '
                           'corresponding to a single feature map.')

    box_predictions = _first_stage_box_predictor_predict([rpn_box_predictor_features], [box_encodings],
                                                         [class_predictions_with_background],
                                                         num_anchors_per_location)

    predictions_box_encodings = np.concatenate(
        box_predictions[BOX_ENCODINGS], axis=1)

    rpn_box_encodings = np.squeeze(predictions_box_encodings, axis=2)

    print("rpn_box_encodings:", rpn_box_encodings.shape)
    print("rpn_box_encodings 0 :", rpn_box_encodings[0, 0, :])

    rpn_objectness_predictions_with_background = np.concatenate(
        box_predictions[CLASS_PREDICTIONS_WITH_BACKGROUND], axis=1)

    # The Faster R-CNN paper recommends pruning anchors that venture outside
    # the image window at training time and clipping at inference time.
    clip_window = np.stack([0, 0, image_shape[1], image_shape[2]])

    feature_map_shape = rpn_features_to_crop.shape

    anchors_boxlist = box_list_ops.concatenate(
        first_stage_anchor_generator.generate([(feature_map_shape[1],
                                                feature_map_shape[2])]))
    # [   0    0  600 1002]
    anchors_boxlist = box_list_ops.clip_to_window(
        anchors_boxlist, clip_window)
    # clip anchors[0]: [ 0.        0.       45.254834 22.627417]
    _anchors = anchors_boxlist

    cropped_regions = _predict_second_stage_1(rpn_box_encodings, rpn_objectness_predictions_with_background,
                                              rpn_features_to_crop, _anchors.get(), image_shape)

    return cropped_regions


def _add_output_tensor_nodes(postprocessed_tensors):
    detection_fields = fields.DetectionResultFields
    label_id_offset = 1

    boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
    scores = postprocessed_tensors.get(detection_fields.detection_scores)
    classes = postprocessed_tensors.get(
        detection_fields.detection_classes) + label_id_offset
    keypoints = postprocessed_tensors.get(detection_fields.detection_keypoints)
    masks = postprocessed_tensors.get(detection_fields.detection_masks)
    # TODO  fixed
    num_detections = postprocessed_tensors.get(detection_fields.num_detections)

    if isinstance(num_detections,  list):
        num_detections = num_detections[0]
    elif isinstance(num_detections, float):
        num_detections = int(num_detections)
    elif isinstance(num_detections, np.ndarray):
        num_detections = int(num_detections[0])

    print("=============== num_detections :", num_detections)
    outputs = {}
    scores = scores.flatten()

    scores_1 = scores[0:num_detections]
    print("scores_1:", scores_1)

    # todo 读取配置文件 置 0 置 1 操作原始代码
    if scores.shape[0] < 100:
        raw_shape = 100
    else:
        raw_shape = scores.shape[0]

    scores_2 = np.zeros(shape=raw_shape - num_detections)
    scores = np.hstack((scores_1, scores_2))
    scores = np.expand_dims(scores, axis=0)

    outputs[detection_fields.detection_scores] = scores

    classes = classes.flatten()

    classes_1 = classes[0:num_detections]
    print("classes_1:", classes_1)
    classes_2 = np.ones(shape=raw_shape - num_detections)
    classes = np.hstack((classes_1, classes_2))
    classes = np.expand_dims(classes, axis=0)

    outputs[detection_fields.detection_classes] = classes

    boxes_1 = boxes[:, 0:num_detections]
    print("boxes_1:", boxes_1)
    boxes_2 = np.zeros(shape=(1, raw_shape - num_detections, 4))
    boxes = np.hstack((boxes_1, boxes_2))

    outputs[detection_fields.detection_boxes] = boxes
    outputs[detection_fields.num_detections] = num_detections

    if keypoints is not None:
        outputs[detection_fields.detection_keypoints] = keypoints

    print("================= scores.shape :", scores.shape)
    print("================= boxes.shape :", boxes.shape)
    print("================= classes.shape :", classes.shape)

    return outputs


def _first_stage_box_predictor_predict(image_features, box_encodings, class_predictions_with_backgrounds,
                                       num_predictions_per_locations):
    box_encodings_list = []
    class_predictions_list = []
    num_classes = 1
    num_class_slots = num_classes + 1


    _proposal_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'proposal')

    _box_coder = _proposal_target_assigner.box_coder

    print("_box_coder:", _box_coder)

    _box_code_size = _box_coder.code_size

    print("_box_code_size:", _box_code_size)

    for (image_feature, box_encoding, class_predictions_with_background,
         num_predictions_per_location) in zip(
            image_features, box_encodings, class_predictions_with_backgrounds,
        num_predictions_per_locations):

        combined_feature_map_shape = (shape_utils.combined_static_and_dynamic_shape(image_feature))

        shapes = np.stack([combined_feature_map_shape[0],
                           combined_feature_map_shape[1] * combined_feature_map_shape[2] * num_predictions_per_location,
                           1,
                           _box_code_size])

        box_encoding_reshape = np.reshape(box_encoding, shapes)

        box_encodings_list.append(box_encoding_reshape)

        class_predictions_with_background = np.reshape(
            class_predictions_with_background,
            np.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))

        class_predictions_list.append(class_predictions_with_background)

    print("box_encodings_list:", np.array(box_encodings_list).shape)
    print("class_predictions_list:", np.array(class_predictions_list).shape)
    return {
        BOX_ENCODINGS: box_encodings_list,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list
    }


def _predict_second_stage_1(rpn_box_encodings,
                            rpn_objectness_predictions_with_background,
                            rpn_features_to_crop,
                            anchors,
                            image_shape):
    image_shape_2d = _image_batch_shape_2d(image_shape)

    first_stage_max_proposals = config.cfg.POSTPROCESSOR.FIRST_STAGE_MAX_PROPOSALS

    proposal_boxes_normalized, _, num_proposals = _postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        anchors, image_shape_2d, first_stage_max_proposals=first_stage_max_proposals)

    cropped_regions = (
        _compute_second_stage_input_feature_maps(
            rpn_features_to_crop, proposal_boxes_normalized))

    return cropped_regions


def _flatten_first_two_dimensions(inputs):
    combined_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
    flattened_shape = np.stack([combined_shape[0] * combined_shape[1]] +
                               combined_shape[2:])
    return np.reshape(inputs, flattened_shape)


def _compute_second_stage_input_feature_maps(features_to_crop,
                                             proposal_boxes_normalized):
    def get_box_inds(proposals):
        proposals_shape = proposals.shape
        ones_mat = np.ones(proposals_shape[:2], dtype=np.int32)
        multiplier = np.expand_dims(range(proposals_shape[0]), 1)
        return np.reshape(ones_mat * multiplier, [-1])

    _initial_crop_size = config.cfg.POSTPROCESSOR.INITIAL_CROP_SIZE

    box_index = get_box_inds(proposal_boxes_normalized)

    # TODO
    import tensorflow as tf
    with tf.Session() as sess:
        cropped_regions = tf.image.crop_and_resize(
            features_to_crop,
            _flatten_first_two_dimensions(proposal_boxes_normalized),
            box_index,
            [_initial_crop_size, _initial_crop_size])

        ksize = config.cfg.POSTPROCESSOR.MAXPOOL_KERNEL_SIZE
        strides = config.cfg.POSTPROCESSOR.MAXPOOL_STRIDE
        slim = tf.contrib.slim

        max_pooled = slim.max_pool2d(
            cropped_regions,
            [ksize, ksize],
            stride=strides)

        max_pooled = sess.run(max_pooled)
        print("============== ROI Pooling result :", max_pooled.shape)

    return max_pooled


def _image_batch_shape_2d(image_batch_shape_1d):
    return np.tile(np.expand_dims(image_batch_shape_1d[1:], 0),
                   [image_batch_shape_1d[0], 1])


def _postprocess_rpn(
        rpn_box_encodings_batch,
        rpn_objectness_predictions_with_background_batch,
        anchors,
        image_shapes, first_stage_max_proposals):
    first_stage_nms_score_threshold = config.cfg.POSTPROCESSOR.FIRST_STAGE_NMS_SCORE_THRESHOLD
    first_stage_nms_iou_threshold = config.cfg.POSTPROCESSOR.FIRST_STAGE_NMS_IOU_THRESHOLD

    rpn_box_encodings_batch = np.expand_dims(rpn_box_encodings_batch, axis=2)

    rpn_encodings_shape = shape_utils.combined_static_and_dynamic_shape(
        rpn_box_encodings_batch)

    print("=== anchors:", anchors[0])

    tiled_anchor_boxes = np.tile(
        np.expand_dims(anchors, 0), [rpn_encodings_shape[0], 1, 1])
    print("=== tiled_anchor_boxes:", tiled_anchor_boxes[0][0])

    proposal_boxes = _batch_decode_boxes(rpn_box_encodings_batch,
                                         tiled_anchor_boxes)

    proposal_boxes = np.squeeze(proposal_boxes, axis=2)

    print("=========== proposal_boxes:", proposal_boxes.shape)
    print("=========== proposal_boxes 0:", proposal_boxes[0, 0])

    rpn_objectness_softmax_without_background = ops.softmax(rpn_objectness_predictions_with_background_batch)[:, :, 1]
    clip_window = _compute_clip_window(image_shapes)

    (proposal_boxes, proposal_scores, _, _, _,
     num_proposals) = post_processing.batch_multiclass_non_max_suppression(
        np.expand_dims(proposal_boxes, axis=2),
        np.expand_dims(rpn_objectness_softmax_without_background,
                       axis=2),
        first_stage_nms_score_threshold,
        first_stage_nms_iou_threshold,
        first_stage_max_proposals,
        first_stage_max_proposals,
        clip_window=clip_window)

    # normalize proposal boxes
    def normalize_boxes(args):
        proposal_boxes_per_image = args[0][0]
        image_shape = args[1][0]

        normalized_boxes_per_image = box_list_ops.to_normalized_coordinates(
            box_list.BoxList(proposal_boxes_per_image), image_shape[0],
            image_shape[1]).get()

        return normalized_boxes_per_image

    normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
        normalize_boxes, elems=[proposal_boxes, image_shapes])

    return normalized_proposal_boxes, proposal_scores, num_proposals


def _compute_clip_window(image_shapes):
    clip_heights = image_shapes[:, 0]
    clip_widths = image_shapes[:, 1]
    clip_window = np.stack([np.zeros_like(clip_heights),
                            np.zeros_like(clip_heights),
                            clip_heights, clip_widths], axis=1)
    return clip_window


def _batch_decode_boxes(box_encodings, anchor_boxes):
    print("================ _batch_decode_boxes ==================")
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)

    num_classes = combined_shape[2]
    # num_classes = 1
    # (1, 28728, 4)
    print("====== anchor_boxes:", anchor_boxes[0][0])
    anchor_boxes_exp_dim = np.expand_dims(anchor_boxes, 2)
    # anchor_boxes_exp_dim: (1, 28728, 1, 4)
    tiled_anchor_boxes = np.tile(anchor_boxes_exp_dim, [1, 1, num_classes, 1])
    # tiled_anchor_boxes (1, 28728, 1, 4)
    reshaped_anchors = np.reshape(tiled_anchor_boxes, [-1, 4])
    # reshaped: (28728, 4)
    print("====== reshaped_anchors:", reshaped_anchors[0])
    tiled_anchors_boxlist = box_list.BoxList(reshaped_anchors)

    _proposal_target_assigner = target_assigner.create_target_assigner(
        'FasterRCNN', 'proposal')

    _box_coder = _proposal_target_assigner.box_coder

    print("================ _box_coder.decode ==================")

    reshaped = np.reshape(box_encodings, [-1, _box_coder.code_size])
    print("reshaped:", reshaped.shape)

    decoded_boxes = _box_coder.decode(reshaped, tiled_anchors_boxlist)

    decoded_boxes_reahpe = np.reshape(decoded_boxes.get(),
                                      np.stack([combined_shape[0], combined_shape[1],
                                                num_classes, 4]))

    print("decoded_boxes_reahpe:", decoded_boxes_reahpe.shape)
    return decoded_boxes_reahpe


def show_detection_result(result):
    print("PATH_TO_LABELS:", PATH_TO_LABELS)
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # NUM_CLASSES
    NUM_CLASSES = config.cfg.POSTPROCESSOR.NUM_CLASSES
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    result['detection_classes'] = result[
        'detection_classes'][0].astype(np.uint8)
    result['detection_boxes'] = result['detection_boxes'][0]
    result['detection_scores'] = result['detection_scores'][0]

    img_dir = config.cfg.PREPROCESS.IMG_LIST
    file_list = os.listdir(img_dir)

    IMG_PATH = os.path.join(img_dir, file_list[0])
    print("IMG_PATH:", IMG_PATH)
    image = Image.open(IMG_PATH)
    image_np = load_image_into_numpy_array(image)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        result['detection_boxes'],
        result['detection_classes'],
        result['detection_scores'],
        category_index,
        instance_masks=result.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    misc.imsave('detection_result_faster_rcnn.png', image_np)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
