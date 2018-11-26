from ..processor import model_config as config
from .utils import standard_fields as fields

from .tensorflow_custom_layer  import  TensorflowProposal
from .tensorflow_custom_layer  import  TensorflowROIPooling

BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'

    rpn_objectness_predictions_with_background = tf.concat(
        box_predictions[CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)
    print("--------------------rpn_objectness_predictions_with_background :",rpn_objectness_predictions_with_background.shape)
    print("-----------------predictions_box_encodings: ",predictions_box_encodings.shape)

    first_stage_anchor_generator = anchor_generator_builder.build("grid_anchor_generator")


    rpn_box_encodings_batch = tf.expand_dims(rpn_box_encodings_batch, axis=2)

    print("rpn_box_encodings_batch name:", rpn_box_encodings_batch.name)
    print("rpn_box_encodings_batch: shape", rpn_box_encodings_batch.shape)
    rpn_encodings_shape = shape_utils.combined_static_and_dynamic_shape(
        rpn_box_encodings_batch)
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(anchors, 0), [rpn_encodings_shape[0], 1, 1])
    print("_batch_decode_boxes 1")
    proposal_boxes = _batch_decode_boxes(rpn_box_encodings_batch,
                                         tiled_anchor_boxes)
    proposal_boxes1 = tf.squeeze(proposal_boxes, axis=2)
    ret_proposal_boxes = proposal_boxes1;

    proposal_boxes = tf.squeeze(proposal_boxes, axis=2)
    rpn_objectness_softmax_without_background = tf.nn.softmax(
        rpn_objectness_predictions_with_background_batch)[:, :, 1]
    clip_window = _compute_clip_window(image_shapes)

    (proposal_boxes, proposal_scores, _, _, _,
     num_proposals) = post_processing.batch_multiclass_non_max_suppression(
        tf.expand_dims(proposal_boxes, axis=2),
        tf.expand_dims(proposal_boxes1, axis=2),
        tf.expand_dims(rpn_objectness_softmax_without_background,
                       axis=2),
        first_stage_nms_score_threshold,

    normalized_proposal_boxes = shape_utils.static_or_dynamic_map_fn(
        normalize_boxes, elems=[proposal_boxes, image_shapes], dtype=tf.float32)
    return normalized_proposal_boxes, proposal_scores, num_proposals

    return normalized_proposal_boxes, proposal_scores, num_proposals,ret_proposal_boxes 


def _compute_clip_window(image_shapes):
                            image_shape):
    image_shape_2d = _image_batch_shape_2d(image_shape)

    proposal_boxes_normalized, _, num_proposals = _postprocess_rpn(
    proposal_boxes_normalized, _, num_proposals,_ = _postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        anchors, image_shape_2d, first_stage_max_proposals=100)

                               combined_shape[2:])
    return tf.reshape(inputs, flattened_shape)

def Proposal_layer(preprocessed_inputs,box_encodings, class_prediction):
    image_shape = preprocessed_inputs.shape
    #clip_window = tf.to_float(tf.stack([0, 0, image_shape[1], image_shape[2]]))

    inputs = [
        class_prediction,
        box_encodings,
	image_shape
    ]
    tfproposal = TensorflowProposal("TensorflowProposal")
    tfproposal.from_config(  feat_stride = 16, scales = (4,8, 16, 32) )
    proposals,boxdecode ,anchors= tfproposal.predict(inputs)
    return proposals,boxdecode,anchors


def ROIPooling_layer(rpn_box_predictor_features,proposals):

    inputs = [
        rpn_box_predictor_features,
        proposals
    ]
    roipooling = TensorflowROIPooling("TensorflowROIPooling")
    roipooling.from_config( pooled_w = 7, pooled_h = 7, spatial_scale = 0.0625 )
    roi = roipooling.predict(inputs)
    return roi 

# rpn_features_to_crop - FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu
# rpn_box_predictor_features Conv/Relu6
    anchors_boxlist = box_list_ops.concatenate(
        first_stage_anchor_generator.generate([(feature_map_shape[1],
                                                feature_map_shape[2])]))
    print("1myanchors....................",anchors_boxlist.get())
    anchors_boxlist = box_list_ops.clip_to_window(
        anchors_boxlist, clip_window)
    _anchors = anchors_boxlist

    print("second_stage_box_predictor _postprocess_rpn")

    myanchors = _anchors.get()
    print("2myanchors....................",myanchors)
    image_shape_2d = _image_batch_shape_2d(image_shape)

    num_anchors_per_location = (
                                                         [rpn_objectness_predictions_with_background],
                                                         num_anchors_per_location)

 
    predictions_box_encodings = tf.concat(
        box_predictions[BOX_ENCODINGS], axis=1)

    print("squeeze predictions_box_encodings.shape:", predictions_box_encodings.shape)

    rpn_box_encodings = tf.squeeze(predictions_box_encodings, axis=2)

    print("rpn_box_encodings.shape:", rpn_box_encodings.shape)

    rpn_objectness_predictions_with_background = tf.concat(
        box_predictions[CLASS_PREDICTIONS_WITH_BACKGROUND],
        axis=1)

    proposal_boxes_normalized, _, num_proposals = _postprocess_rpn(
    proposal_boxes_normalized, _, num_proposals,ret = _postprocess_rpn(
        rpn_box_encodings, rpn_objectness_predictions_with_background,
        _anchors.get(), image_shape_2d, first_stage_max_proposals=100)


    result_output = second_postprocess(prediction_dict, true_image_shapes)

    return result_output
    return result_output,ret,myanchors


