import numpy as np
from platformx.plat_tensorflow.tools.processor.np_utils import shape_utils, \
    anchor_generator_builder, box_list_ops, box_list, box_coder_builder, post_processing_builder, \
    visualization_utils as vis_util
from platformx.plat_tensorflow.tools.processor.np_utils import standard_fields as fields
from platformx.plat_tensorflow.tools.processor import model_config
import config
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from platformx.plat_tensorflow.tools.processor.np_utils import label_map_util
from scipy import misc
import os

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'
BASE_BoxEncodingPredictor = "_BoxEncodingPredictor"
BASE_ClassPredictor = "_ClassPredictor"
PPN_BoxPredictor_0 = "WeightSharedConvolutionalBoxPredictor_BoxPredictor"
PPN_ClassPredictor_0 = "WeightSharedConvolutionalBoxPredictor_ClassPredictor"
BASE_PPN_BoxPredictor = "_BoxPredictor"
BASE_PPN_ClassPredictor = "WeightSharedConvolutionalBoxPredictor"
PATH_TO_LABELS = config.cfg.POSTPROCESSOR.PATH_TO_LABELS

def run_ssd_tf_post(preprocessed_inputs, result_middle=None):
    boxes_encodings_np = []
    classes_predictions_with_background_np = []
    feature_maps_np = []

    for i in range(6):
        for key, value in result_middle.items():
            if str(i) + BASE_BoxEncodingPredictor in key:
                print(str(i) + BASE_BoxEncodingPredictor + ": ", value.shape)
                boxes_encodings_np.append(value)
                break
            if i == 0:
                if PPN_BoxPredictor_0 in key:
                    print("PPN_BoxPredictor_0:", value.shape)
                    boxes_encodings_np.append(value)
                    break
            else:
                if str(i) + BASE_PPN_BoxPredictor in key:
                    print(str(i) + BASE_PPN_BoxPredictor, value.shape)
                    boxes_encodings_np.append(value)
                    break

        for key, value in result_middle.items():
            if str(i) + BASE_ClassPredictor in key and BASE_PPN_ClassPredictor not in key:
                print(str(i) + BASE_ClassPredictor+ ": ", value.shape)
                classes_predictions_with_background_np.append(value)
                break

            if i == 0:
                if PPN_ClassPredictor_0 in key:
                    print(PPN_ClassPredictor_0 + ":", value.shape)
                    classes_predictions_with_background_np.append(value)
                    break
            else:
                if str(i) + BASE_ClassPredictor in key and BASE_PPN_ClassPredictor in key:
                    print(str(i) + BASE_ClassPredictor + ":", value.shape)
                    classes_predictions_with_background_np.append(value)
                    break

    for key, value in result_middle.items():
        if "FeatureExtractor" in key and "fpn" not in key:
            print("key {}  value {}".format(key, value.shape))
            feature_maps_np.append(value)

    if len(feature_maps_np) < 1:
        key_dict = {}
        for key, value in result_middle.items():
            if "FeatureExtractor" in key and "fpn"in key:
                key_dict[key] = value.shape[1]

        sorted_key_dict = sorted(key_dict.items(), key=lambda x: x[1], reverse=True)
        for key, value in sorted_key_dict:
            feature_maps_np.append(result_middle[key])

    input_shape = preprocessed_inputs.shape
    true_image_shapes = np.array([input_shape[1], input_shape[2], input_shape[3]], dtype=np.int32)
    true_image_shapes = true_image_shapes.reshape((1, 3))
    post_result = post_deal(boxes_encodings_np, classes_predictions_with_background_np, feature_maps_np,
                            preprocessed_inputs,
                            true_image_shapes)

    show_detection_result(post_result)

    return post_result


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

    # IMAGE_SIZE = (12, 8)
    # plt.figure(figsize=IMAGE_SIZE)
    misc.imsave('detection_result_ssd.png', image_np)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def post_deal(boxes_encodings, classes_predictions_with_background, feature_maps, preprocessed_inputs=None,
              true_image_shapes=None):
    """
    SSD model POST processer

    :param boxes_encodings:
    :param classes_predictions_with_background:
    :param feature_maps:
    :param preprocessed_inputs:
    :param true_image_shapes:
    :return:
    """
    prediction_dict, anchors = last_predict_part(boxes_encodings, classes_predictions_with_background, feature_maps,
                                                 preprocessed_inputs)
    postprocessed_tensors = postprocess(anchors, prediction_dict, true_image_shapes)

    return _add_output_tensor_nodes(postprocessed_tensors)


def _add_output_tensor_nodes(postprocessed_tensors):
    print("------------------ _add_output_tensor_nodes ------------------")
    detection_fields = fields.DetectionResultFields
    label_id_offset = 1

    boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
    scores = postprocessed_tensors.get(detection_fields.detection_scores)
    classes = postprocessed_tensors.get(
        detection_fields.detection_classes) + label_id_offset
    keypoints = postprocessed_tensors.get(detection_fields.detection_keypoints)
    masks = postprocessed_tensors.get(detection_fields.detection_masks)
    num_detections = postprocessed_tensors.get(detection_fields.num_detections)

    if isinstance(num_detections, list):
        num_detections = num_detections[0]
    elif isinstance(num_detections, float):
        num_detections = int(num_detections)
    elif isinstance(num_detections, np.ndarray):
        num_detections = int(num_detections[0])

    print("=============== num_detections :", num_detections)
    outputs = {}
    print("scores:", scores)
    scores = scores.flatten()
    # todo 读取配置文件 置 0 置 1 操作原始代码
    if scores.shape[0] < 100:
        raw_shape = 100
    else:
        raw_shape = scores.shape[0]

    scores_1 = scores[0:num_detections]
    print("scores_1:", scores_1)
    scores_2 = np.zeros(shape=raw_shape - num_detections)
    scores = np.hstack((scores_1, scores_2))
    scores = np.reshape(scores, (1, scores.shape[0]))

    outputs[detection_fields.detection_scores] = scores

    classes = classes.flatten()
    classes_1 = classes[0:num_detections]
    print("classes_1:", classes_1)
    classes_2 = np.ones(shape=raw_shape - num_detections)
    classes = np.hstack((classes_1, classes_2))
    classes = np.reshape(classes, (1, classes.shape[0]))

    outputs[detection_fields.detection_classes] = classes

    boxes_1 = boxes[:, 0:num_detections]
    print("boxes_1:", boxes_1)
    boxes_2 = np.zeros(shape=(1, raw_shape - num_detections, 4))
    boxes = np.hstack((boxes_1, boxes_2))

    outputs[detection_fields.detection_boxes] = boxes
    outputs[detection_fields.num_detections] = num_detections

    if keypoints is not None:
        outputs[detection_fields.detection_keypoints] = keypoints
    if masks is not None:
        outputs[detection_fields.detection_masks] = masks

    return outputs


def last_predict_part(boxes_encodings, classes_predictions_with_background, feature_maps, preprocessed_inputs=None):
    print("------------------ last_predict_part ------------------")
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the forward
    pass of the network to yield unpostprocessesed predictions.

    A side effect of calling the predict method is that self._anchors is
    populated with a box_list.BoxList of anchors.  These anchors must be
    constructed before the postprocess or loss functions can be called.

    Args:
      boxes_encodings:
      classes_predictions_with_background:
      feature_maps:

      preprocessed_inputs: a [batch, height, width, channels] image tensor.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    """
    anchor_generator = anchor_generator_builder.build()

    num_predictions_per_location_list = anchor_generator.num_anchors_per_location()

    # print("num_predictions_per_location_list:", num_predictions_per_location_list)
    prediction_dict = post_processor(boxes_encodings, classes_predictions_with_background,
                                     feature_maps, num_predictions_per_location_list)

    image_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_inputs)
    feature_map_spatial_dims = get_feature_map_spatial_dims(
        feature_maps)

    anchors_list = anchor_generator.generate(
        feature_map_spatial_dims,
        im_height=image_shape[1],
        im_width=image_shape[2])

    anchors = box_list_ops.concatenate(anchors_list)

    box_encodings = np.concatenate(prediction_dict['box_encodings'], axis=1)
    if box_encodings.ndim == 4 and box_encodings.shape[2] == 1:
        box_encodings = np.squeeze(box_encodings, axis=2)

    class_predictions_with_background = np.concatenate(
        prediction_dict['class_predictions_with_background'], axis=1)
    predictions_dict = {
        'preprocessed_inputs': preprocessed_inputs,
        'box_encodings': box_encodings,
        'class_predictions_with_background':
            class_predictions_with_background,
        'feature_maps': feature_maps,
        'anchors': anchors.get()
    }
    return predictions_dict, anchors


def get_feature_map_spatial_dims(feature_maps):
    """Return list of spatial dimensions for each feature map in a list.

    Args:
      feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].

    Returns:
      a list of pairs (height, width) for each feature map in feature_maps
    """
    feature_map_shapes = [
        shape_utils.combined_static_and_dynamic_shape(
            feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]


def post_processor(boxes_encodings, classes_predictions_with_background, image_features,
                   num_predictions_per_location_list):

    print("------------------ post_processor ------------------")

    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A list of float tensors of shape [batch_size, height_i,
        width_i, channels_i] containing features for a batch of images.
      num_predictions_per_location_list: A list of integers representing the
        number of box predictions to be made per spatial location for each
        feature map.

    Returns:
      box_encodings: A list of float tensors of shape
        [batch_size, num_anchors_i, q, code_size] representing the location of
        the objects, where q is 1 or the number of classes. Each entry in the
        list corresponds to a feature map in the input `image_features` list.
      class_predictions_with_background: A list of float tensors of shape
        [batch_size, num_anchors_i, num_classes + 1] representing the class
        predictions for the proposals. Each entry in the list corresponds to a
        feature map in the input `image_features` list.
    """

    box_encodings_list = []
    class_predictions_list = []

    for (image_feature,
         num_predictions_per_location,
         box_encodings,
         class_predictions_with_background) in zip(image_features,
                                                   num_predictions_per_location_list,
                                                   boxes_encodings,
                                                   classes_predictions_with_background):
        combined_feature_map_shape = image_feature.shape

        box_code_size = config.cfg.POSTPROCESSOR.BOX_CODE_SIZE

        new_shape = np.stack([combined_feature_map_shape[0],
                                     combined_feature_map_shape[1] *
                                     combined_feature_map_shape[2] *
                                     num_predictions_per_location,
                                     1, box_code_size])

        box_encodings = np.reshape(box_encodings, new_shape)
        box_encodings_list.append(box_encodings)
        num_classes = config.cfg.POSTPROCESSOR.NUM_CLASSES
        num_class_slots = num_classes + 1

        class_predictions_with_background = np.reshape(
            class_predictions_with_background,
            np.stack([combined_feature_map_shape[0],
                      combined_feature_map_shape[1] *
                      combined_feature_map_shape[2] *
                      num_predictions_per_location,
                      num_class_slots]))
        class_predictions_list.append(class_predictions_with_background)

    return {BOX_ENCODINGS: box_encodings_list,
            CLASS_PREDICTIONS_WITH_BACKGROUND: class_predictions_list}


def postprocess(anchors, prediction_dict, true_image_shapes):

    print("------------------ postprocess ------------------")
    if ('box_encodings' not in prediction_dict or
            'class_predictions_with_background' not in prediction_dict):
        raise ValueError('prediction_dict does not contain expected entries.')

    preprocessed_images = prediction_dict['preprocessed_inputs']
    box_encodings = prediction_dict['box_encodings']
    box_encodings = box_encodings
    class_predictions = prediction_dict['class_predictions_with_background']
    detection_boxes, detection_keypoints = _batch_decode(anchors, box_encodings)
    detection_boxes = detection_boxes
    detection_boxes = np.expand_dims(detection_boxes, axis=2)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(model_config.SSD)
    detection_scores_with_background = score_conversion_fn(class_predictions)
    detection_scores = detection_scores_with_background[0:, 0:, 1:]
    additional_fields = None
    if detection_keypoints is not None:
        additional_fields = {
            fields.BoxListFields.keypoints: detection_keypoints}

    (nmsed_boxes, nmsed_scores, nmsed_classes, _, nmsed_additional_fields,
     num_detections) = non_max_suppression_fn(
        detection_boxes,
        detection_scores,
        clip_window=_compute_clip_window(
            preprocessed_images, true_image_shapes),
        additional_fields=additional_fields)

    detection_dict = {
        fields.DetectionResultFields.detection_boxes: nmsed_boxes,
        fields.DetectionResultFields.detection_scores: nmsed_scores,
        fields.DetectionResultFields.detection_classes: nmsed_classes,
        fields.DetectionResultFields.num_detections:
            float(num_detections)
    }

    if (nmsed_additional_fields is not None and
            fields.BoxListFields.keypoints in nmsed_additional_fields):
        detection_dict[fields.DetectionResultFields.detection_keypoints] = (
            nmsed_additional_fields[fields.BoxListFields.keypoints])

    return detection_dict


def _compute_clip_window(preprocessed_images, true_image_shapes):
    resized_inputs_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_images)
    true_heights, true_widths, _ = np.split(true_image_shapes, 3, axis=1)
    padded_height = float(resized_inputs_shape[1])
    padded_width = float(resized_inputs_shape[2])

    cliped_image = np.stack(
        [np.zeros_like(true_heights), np.zeros_like(true_widths),
         true_heights / padded_height, true_widths / padded_width], axis=1)

    cliped_imaged = cliped_image.reshape(1, -1)
    return cliped_imaged


def _batch_decode(anchors, box_encodings):
    """Decodes a batch of box encodings with respect to the anchors.

    Args:
      box_encodings: A float32 tensor of shape
        [batch_size, num_anchors, box_code_size] containing box encodings.

    Returns:
      decoded_boxes: A float32 tensor of shape
        [batch_size, num_anchors, 4] containing the decoded boxes.
      decoded_keypoints: A float32 tensor of shape
        [batch_size, num_anchors, num_keypoints, 2] containing the decoded
        keypoints if present in the input `box_encodings`, None otherwise.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = np.tile(
        np.expand_dims(anchors.get(), 0), [batch_size, 1, 1])

    tiled_anchors_boxlist = box_list.BoxList(
        np.reshape(tiled_anchor_boxes, [-1, 4]))

    box_coder = box_coder_builder.build("faster_rcnn_box_coder")
    decoded_boxes = box_coder.decode(
        np.reshape(box_encodings, [-1, box_coder.code_size]),
        tiled_anchors_boxlist)

    decoded_keypoints = None
    if decoded_boxes.has_field(fields.BoxListFields.keypoints):
        decoded_keypoints = decoded_boxes.get_field(
            fields.BoxListFields.keypoints)
        num_keypoints = decoded_keypoints.get_shape()[1]
        decoded_keypoints = np.reshape(
            decoded_keypoints,
            np.stack([combined_shape[0], combined_shape[1], num_keypoints, 2]))
    decoded_boxes = np.reshape(decoded_boxes.get(), np.stack(
        [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes, decoded_keypoints

