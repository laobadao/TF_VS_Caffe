# -*- coding: utf-8 -*-
from __future__ import absolute_import

from platformx.plat_tensorflow.tools.processor.np_utils import visualization_utils as vis_util
from platformx.plat_tensorflow.tools.processor.np_utils import label_map_util
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import h5py
import platformx.plat_tensorflow.tools.processor.np_faster_rcnn_post as np_frp
from tensorflow.core.framework import graph_pb2

IMG_PATH = "data\\detection_500_1000.jpg"
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

np.set_printoptions(threshold=np.inf)

MIDDLE_OUT = None


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_beach():
    """Get beach image as input."""

    name = "image_tensor:0"
    # image = Image.open(IMG_PATH)
    # img_np = load_image_into_numpy_array(image)
    # img_np = np.expand_dims(img_np, axis=0)
    #
    import cv2
    img = cv2.imread(IMG_PATH)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (im_height, im_width) = image.shape[0], image.shape[1]
    image_np = image.reshape((im_height, im_width, 3)).astype(np.uint8)
    image_np = np.expand_dims(image_np, axis=0)
    print("image_np:", image_np.shape)
    return {name: image_np}


def run_tensorflow(sess, inputs, test_name):
    # print('run_tensorflow(): so we have a reference output')
    """Run model on tensorflow so we have a referecne output."""
    feed_dict = {}
    for k, v in inputs.items():
        k = sess.graph.get_tensor_by_name(k)
        feed_dict[k] = v

    out_tensor_input = ["Preprocessor/sub:0"]

    out_tensor_middle = ['FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd:0',
                         'FirstStageBoxPredictor/ClassPredictor/BiasAdd:0',
                         'Conv/Relu6:0',
                         'FirstStageFeatureExtractor/resnet_v1_50/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0',
                         "SecondStageBoxPredictor/Reshape:0",
                         "SecondStageBoxPredictor/Reshape_1:0"]

    out_tensor_final = ["detection_boxes:0",
                        "detection_scores:0",
                        "num_detections:0",
                        "detection_classes:0"]

    roi_polling_tensor = ["MaxPool2D/MaxPool:0"]

    if test_name:
        test = [test_name]
        test_output = sess.run(test, feed_dict=feed_dict)
        print("test_output shape:", test_output[0].shape)

    result_input = sess.run(out_tensor_input, feed_dict=feed_dict)

    result_middle = None
    result_middle = sess.run(out_tensor_middle, feed_dict=feed_dict)

    roi_pooling_out = sess.run(roi_polling_tensor, feed_dict=feed_dict)

    print("crop_and_resize_out shape:", roi_pooling_out[0].shape)

    result_final = sess.run(out_tensor_final, feed_dict=feed_dict)

    output_dict = {}
    #
    # for i in range(len(result_final)):
    #     print("result_final shape:", result_final[i].shape)
    #
    output_dict["detection_boxes"] = result_final[0]
    output_dict["detection_scores"] = result_final[1]
    output_dict["num_detections"] = result_final[2]
    output_dict["detection_classes"] = result_final[3]

    # print("================ tf detection_scores", output_dict["detection_scores"])
    #
    # print("================ tf detection_boxes", output_dict["detection_boxes"])
    #
    # print("================ tf detection_classes", output_dict["detection_classes"])

    # print("raw tf pb run done")
    return result_input, result_middle, roi_pooling_out, output_dict


def show_detection_result(result, name):
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # NUM_CLASSES
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    result['num_detections'] = int(result['num_detections'][0])
    result['detection_classes'] = result[
        'detection_classes'][0].astype(np.uint8)
    result['detection_boxes'] = result['detection_boxes'][0]
    result['detection_scores'] = result['detection_scores'][0]

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

    IMAGE_SIZE = (12, 8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    #
    from scipy import misc
    misc.imsave(name + '_detection_result.png', image_np)
    plt.show()


def main(arg=None, arg1=None):
    model_path = r"E:\Intenginetech\ConvertTool\python\platformx\plat_tensorflow\tests\models\faster_rcnn_resnet50\faster_rcnn_resnet50.pb"
    graph_def = graph_pb2.GraphDef()
    with open(model_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    g = tf.import_graph_def(graph_def, name='')

    inputs = get_beach()
    test_name = None

    with tf.Session(graph=g) as sess:
        # run the model with tensorflow
        result_input, result_middle, crop_and_resize_out, output_dict = run_tensorflow(sess, inputs, test_name)

    show_detection_result(output_dict, 'raw')

    # TODO numpy method

    # fisrt_post_result = np_frp.faster_rcnn_stage_one_post(result_input, result_middle)
    #
    # print("fisrt_post_result:", fisrt_post_result.shape)
    #
    # f = h5py.File('fisrt_post_result.h5', 'w')
    # f["fisrt_post_result"] = fisrt_post_result

    # f = h5py.File('result_input.h5', 'w')
    # f["result_input"] = result_input
    # f = h5py.File('result_middle_0.h5', 'w')
    # f["result_middle_0"] = result_middle[0]
    # f = h5py.File('result_middle_1.h5', 'w')
    # f["result_middle_1"] = result_middle[1]
    # f = h5py.File('result_middle_2.h5', 'w')
    # f["result_middle_2"] = result_middle[2]
    # f = h5py.File('result_middle_3.h5', 'w')
    # f["result_middle_3"] = result_middle[3]
    # f = h5py.File('result_middle_4.h5', 'w')
    # f["result_middle_4"] = result_middle[4]
    # f = h5py.File('result_middle_5.h5', 'w')
    # f["result_middle_5"] = result_middle[5]

    print("check_result Done")


def check_result(result_final, result, test_output_op):
    print("detection_scores:", result["detection_scores"].shape)
    print("detection_scores:", result_final["detection_scores"].shape)
    print("diff:", result_final["detection_scores"] - result["detection_scores"])
    # print("diff one  op out:", test_output_op[0] - MIDDLE_OUT)


def np_faster_post():
    import platform

    BOXES_NAME = "FirstStageBoxPredictor_BoxEncodingPredictor"
    CLASSES_NAME = "FirstStageBoxPredictor_ClassPredictor"
    FEATURES_NAME = "FirstStageFeatureExtractor"
    BOX_PREDICTOR = "SecondStageBoxPredictor_Reshape_1"

    result_middle1_dict = {}
    result_middle2_dict = {}

    if platform.system() == "Windows":
        h0 = 'result_middle_0.h5'
        h1 = 'result_middle_1.h5'
        h2 = 'result_middle_2.h5'
        h3 = 'result_middle_3.h5'
        h4 = 'result_middle_4.h5'
        h5 = 'result_middle_5.h5'
        image_imput = 'result_input.h5'

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

    result_middle2_dict["SecondStageBoxPredictor/Reshape"] = result_middle4
    result_middle2_dict[BOX_PREDICTOR] = result_middle5

    roi_result = np_frp.faster_rcnn_stage_one_post(result_input[0], result_middle1_dict)

    print("roi_result:", roi_result.shape)
    print("roi_result 0:", roi_result[0, :, :, 0])
    # print("roi_result 1:", roi_result[1, :, :, 0])

    # second_post_result = np_frp.faster_rcnn_second_stage_post(result_input, result_middle1_dict, result_middle2_dict)


if __name__ == "__main__":
    # test_name = None
    # h5_name = None
    # main(test_name, h5_name)
    np_faster_post()
