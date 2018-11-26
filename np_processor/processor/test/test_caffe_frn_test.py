
def Proposal_layer(preprocessed_inputs, box_encodings, class_prediction):
    from platformx.plat_tensorflow.tools.processor.test.test_tf_custom_layer import TensorflowProposal

    image_shape = preprocessed_inputs.shape
    inputs = [
        class_prediction,
        box_encodings,
        image_shape
    ]
    tfproposal = TensorflowProposal("TensorflowProposal")

    tfproposal.from_config(feat_stride=16, scales=(4, 8, 16, 32))

    proposals, boxdecode, anchors = tfproposal.predict(inputs)

    return proposals, boxdecode, anchors


def ROIPooling_layer(rpn_box_predictor_features, proposals):
    from platformx.plat_tensorflow.tools.processor.test.test_tf_custom_layer import TensorflowROIPooling
    inputs = [
        rpn_box_predictor_features,
        proposals
    ]
    roipooling = TensorflowROIPooling("TensorflowROIPooling")
    roipooling.from_config(pooled_w=7, pooled_h=7, spatial_scale=0.0625)
    roi = roipooling.predict(inputs)
    return roi


def run_caffe_frn_post(result_input, result_middle):
    box_encodings = result_middle[0]

    class_predictions_with_background = result_middle[1]

    rpn_box_predictor_features = result_middle[3]

    proposal, boxdecode, anchors = Proposal_layer(result_input, box_encodings, class_predictions_with_background)

    roi_result = ROIPooling_layer(rpn_box_predictor_features, proposal)

    print("roi_result:", roi_result.shape)
    # print("roi_result 0:", roi_result[0, :, :, 0])
    # print("roi_result 1:", roi_result[1, :, :, 0])

    # return roi, proposal, boxdecode, anchors

if __name__ == "__main__":
    import platform
    import h5py

    BOXES_NAME = "FirstStageBoxPredictor_BoxEncodingPredictor"
    CLASSES_NAME = "FirstStageBoxPredictor_ClassPredictor"
    FEATURES_NAME = "FirstStageFeatureExtractor"
    BOX_PREDICTOR = "SecondStageBoxPredictor_Reshape_1"

    result_middle1_dict = {}
    result_middle2_dict = {}

    if platform.system() == "Linux":
        h0 = 'platformx/plat_tensorflow/tools/processor/test/result_middle_0.h5'
        h1 = 'platformx/plat_tensorflow/tools/processor/test/result_middle_1.h5'
        h2 = 'platformx/plat_tensorflow/tools/processor/test/result_middle_2.h5'
        h3 = 'platformx/plat_tensorflow/tools/processor/test/result_middle_3.h5'
        h4 = 'platformx/plat_tensorflow/tools/processor/test/result_middle_4.h5'
        h5 = 'platformx/plat_tensorflow/tools/processor/test/result_middle_5.h5'
        image_imput = 'platformx/plat_tensorflow/tools/processor/test/result_input.h5'

    f = h5py.File(h0, 'r')
    result_middle0 = f["result_middle_0"].value
    f1 = h5py.File(h1, 'r')
    result_middle_1 = f1["result_middle_1"].value
    f2 = h5py.File(h2, 'r')
    result_middle2 = f2["result_middle_2"].value
    f3 = h5py.File(h3, 'r')
    result_middle3 = f3["result_middle_3"].value
    f4 = h5py.File(h4, 'r')
    result_middle4 = f4["result_middle_4"].value
    f5 = h5py.File(h5, 'r')
    result_middle5 = f5["result_middle_5"].value

    f = h5py.File(image_imput, 'r')
    result_input = f["result_input"].value

    result_middle1_dict[BOXES_NAME] = result_middle0
    print("result_middle0:", result_middle0.shape)
    result_middle1_dict[CLASSES_NAME] = result_middle_1
    print("result_middle_1:", result_middle_1.shape)
    result_middle1_dict["Conv_Relu6"] = result_middle2
    print("result_middle2:", result_middle2.shape)
    result_middle1_dict[FEATURES_NAME] = result_middle3
    print("result_middle3:", result_middle3.shape)

    #('result_middle0:', (1, 38, 63, 48)) box_encodings
    # ('result_middle_1:', (1, 38, 63, 24)) class_prediction

    # ('result_middle2:', (1, 38, 63, 512)) rpn_box_feature
    # ('result_middle3:', (1, 38, 63, 1024)) rpn_deature_to_crop

    result_middle = [result_middle0, result_middle_1, result_middle2, result_middle3]

    if platform.system() == "Linux":
        run_caffe_frn_post(result_input[0], result_middle)
