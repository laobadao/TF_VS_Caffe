from matplotlib import pyplot as plt
import os

IMG_PATH = "data\\detection_960_660.jpg"
IMG_PATH = "/data1/home/nntool/lirongfeng/source/ConvertTool/python/platformx/plat_tensorflow/tools/processor/test/data/detection_960_660.jpg"
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

PATH_TO_LABELS = "/data1/home/nntool/lirongfeng/source/ConvertTool/python/platformx/plat_tensorflow/tools/processor/test/data/mscoco_label_map.pbtxt"
NUM_CLASSES = 90

np.set_printoptions(threshold=np.inf)
                        "num_detections:0",
                        "detection_classes:0"]

    crop_and_resize_tensor = ["CropAndResize:0"]
    #crop_and_resize_tensor = ["CropAndResize:0"]
    crop_and_resize_tensor = ["MaxPool2D/MaxPool:0"]

    test = [test_name]



def main():
    model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\faster_rcnn_resnet50\\frozen_inference_graph.pb"
    #model_path = "E:\\Intenginetech\\tf-onnx\\tests\\models\\faster_rcnn_resnet50\\frozen_inference_graph.pb"
    #model_path = "C:\\work\\faster_rcnn_resnet50\\frozen_inference_graph.pb"
    model_path = "/data1/home/nntool/lirongfeng/source/ConvertTool/python/platformx/plat_tensorflow/tests/models/faster_rcnn_resnet50/frozen_inference_graph.pb"
    graph_def = graph_pb2.GraphDef()

    print("model_path:", model_path)
    with tf.Session(graph=g) as sess:
        # run the model with tensorflow
        result_input, result_middle, crop_and_resize_out, output_dict = run_tensorflow(sess, inputs, test_name)

    show_detection_result(output_dict, 'raw')
	
    #show_detection_result(output_dict, 'raw')
    roi,proposal,cfdecode,cfanchors = run_faster_rcnn_tf_post_numpy(result_input[0], result_middle, crop_and_resize_out, test_name)
    print("tensorflow:crop_and_resize_out.shape: ",crop_and_resize_out[0].shape)
    for i in range(1,80):
    	diff = roi[i]-crop_and_resize_out[0][i]
    	#print("caffe:roi.shape- caffe roi top 10: ",i,np.max(np.abs(diff)))
    
    #
    result, test_output_op = run_faster_rcnn_tf_post(result_input, result_middle, crop_and_resize_out, test_name)
    show_detection_result(result, 'modified')

    check_result(output_dict, result, test_output_op)

    #result, test_output_op = run_faster_rcnn_tf_post(result_input, result_middle, crop_and_resize_out, test_name)
    #show_detection_result(result, 'modified')

    #check_result(output_dict, result, test_output_op)
    result, test_output_op ,tfdecode,tfanchors= run_faster_rcnn_tf_post(result_input, result_middle, crop_and_resize_out, test_name)
    #print("tfanchors",tfanchors)
    diff = cfdecode - tfdecode
    #print("dfdecode: ",cfdecode[:20])
    print("-----------------------------")
    #print("tfdecode: ",tfdecode[:20])
    print("dfdecode- tfdecode: ",np.max(np.abs(diff)))
    #print("diff: ",diff)
    print("check_result Done")

def run_faster_rcnn_tf_post_numpy(result_input, result_middle, crop_and_resize_out, test_name):
    tf.reset_default_graph()
   
    box_encodings = result_middle[0]
    class_predictions_with_background = result_middle[1]
    rpn_box_predictor_features = result_middle[3]

    proposal ,boxdecode,anchors = frcnn.Proposal_layer(result_input,box_encodings,class_predictions_with_background)
    #print("xxxxxxxxxxxxxxxxxxxxxxxx caffe anchors.shape",anchors.shape)
    print("xxxxxxxxxxxxxxxxxxxxxxxx caffe anchors",anchors[1])
    print("xxxxxxxxxxxxxxxxxxxxxxxx caffe anchors.shape",anchors.shape)
    roi = frcnn.ROIPooling_layer(rpn_box_predictor_features,proposal)
    print("caffe roi.shape",roi.shape)
    return roi,proposal,boxdecode,anchors

def run_faster_rcnn_tf_post(result_input, result_middle, crop_and_resize_out, test_name):
    tf.reset_default_graph()
    print("result_input.shape:", input_shape)
    preprocessed_inputs_np = result_input[0]
    box_encodings = result_middle[0]


    class_predictions_with_background = result_middle[1]

    rpn_box_predictor_features = result_middle[2]
    print("rpn_box_predictor_features.shape:", rpn_box_predictor_features.shape)
    rpn_features_to_crop = result_middle[3]

    with tf.Session() as sess1:
        box_encoding_reshape = result_middle[4]
        class_prediction_reshape = result_middle[5]

        print("box_encoding_reshape:", box_encoding_reshape.shape)
        print("class_prediction_reshape:", class_prediction_reshape.shape)

        box_encoding_reshape_holder = tf.placeholder(dtype=tf.float32, shape=[box_encoding_reshape.shape[0],
                                                                              box_encoding_reshape.shape[1],
                                                                                  ])

        result_input_shape_np = np.array([input_shape[1], input_shape[2], input_shape[3]], dtype=np.int32)
        print("type(result_input_shape_np):", type(result_input_shape_np))
        result_input_shape_np = result_input_shape_np.reshape((1, 3))
        result_input_shape_holder = tf.placeholder(dtype=tf.int32, shape=[1, 3])

        second_stage_out = frcnn.second_stage_box_predictor(
        second_stage_out,ret,result_anchors = frcnn.second_stage_box_predictor(
                                preprocessed_inputs=result_input_holder,
                                box_encoding_reshape=box_encoding_reshape_holder,
                                class_prediction_reshape=class_prediction_reshape_holder,
                                rpn_box_predictor_features=rpn_box_predictor_features_holder
        )


        feed_dict1[box_encoding_reshape_holder] = box_encoding_reshape
        feed_dict1[class_prediction_reshape_holder] = class_prediction_reshape
        feed_dict1[result_input_shape_holder] = result_input_shape_np

        result2 = sess1.run(second_stage_out, feed_dict=feed_dict1)

        # print("result2:", result2)
        
	result3 = sess1.run(ret, feed_dict=feed_dict1)
	result_anchors = sess1.run(result_anchors, feed_dict=feed_dict1)
	
        result_anchors1 = sess1.run("ClipToWindow_1/Gather/GatherV2:0", feed_dict=feed_dict1)
        result_anchors2 = sess1.run("Concatenate_1/concat:0", feed_dict=feed_dict1)
	#result4 = sess1.run("Squeeze_1:0", feed_dict=feed_dict1)
        #print("3-4",np.max(np.abs(result3-result4) ))
        
        result3 = np.squeeze(result3)
       # print("result2:", result2)

        # print("crop_and_resize_out[0]:", crop_and_resize_out[0][0].shape)
        # print("diff crop_and_resize:", crop_and_resize_out[0][0] - result[0])

        print("result detection_boxes:", result2["detection_boxes"].shape)
        print("result detection_scores:", result2["detection_scores"].shape)
        print("result detection_classes:", result2["detection_classes"].shape)
        print("result num_detections:", result2["num_detections"].shape)

    return result2, test_output_op
	print("+++++++++++++++++++++++++result_anchors",result_anchors[1])
	print("+++++++++++++++++++++++++result_anchors1",result_anchors1[1])
	print("+++++++++++++++++++++++++result_anchors2",result_anchors2[1])
        
	result_anchors4 = sess1.run("strided_slice_9:0", feed_dict=feed_dict1)
	result_anchors5 = sess1.run("strided_slice_10:0", feed_dict=feed_dict1)
	print("+++++++++++++++++++++++++withd",result_anchors4)
	print("+++++++++++++++++++++++++height",result_anchors5)
        result_anchors6 = sess1.run("GridAnchorGenerator/add_1:0", feed_dict=feed_dict1)
        print("result_anchors6",result_anchors6.shape)
        result_anchors7 = sess1.run("GridAnchorGenerator/add:0", feed_dict=feed_dict1)
        print("result_anchors7",result_anchors7.shape)
        result_anchors8 = sess1.run("GridAnchorGenerator/Meshgrid_1/Tile:0", feed_dict=feed_dict1)
        print("result_anchors8",result_anchors8.shape)

    return result2, test_output_op,result3,result_anchors


def check_result(result_final, result, test_output_op):
        self._anchor_scales = scales

    def predict(self, inputs):
        print('input_0.shape=', inputs[0].shape)
        print('input_1.shape=', inputs[1].shape)
        _anchors = generate_anchors(scales=np.array(self._anchor_scales))
        _num_anchors = _anchors.shape[0]

            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]

        print('proposals.shape=', proposals.shape)
        return proposals

    def _filter_boxes(self, boxes, min_size):
        img = inputs[0]
        rois = inputs[1]

        print('img shape = ', img.shape)
        num_rois = len(rois)
        print('------------------img shape = ', img.shape)
        print("-----------------rois = ",rois.shape)
	num_rois = len(rois)
        im_h, im_w = img.shape[1], img.shape[2]

        outputs = np.zeros((num_rois, self._pooled_h, self._pooled_w, img.shape[3]))
                    pooled_val = np.max(np.max(crop, axis=0), axis=0)
                    outputs[i_r, ph, pw, : ] = pooled_val

        print("-----------------outputs= ",outputs.shape)
        return outputs
