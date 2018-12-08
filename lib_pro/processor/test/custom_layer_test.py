import unittest
import numpy as np
from keras import Model
from keras.layers import Input
import h5py
import platform
import tensorflow as tf

np.set_printoptions(threshold=np.inf)


class CustomLayerTests(unittest.TestCase):

    def setUp(self):
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

        f = h5py.File(h0, 'r')
        self.result_middle0 = f["result_middle_0"].value
        f1 = h5py.File(h1, 'r')
        self.result_middle_1 = f1["result_middle_1"].value
        f2 = h5py.File(h2, 'r')
        self.result_middle2 = f2["result_middle_2"].value
        f3 = h5py.File(h3, 'r')
        self.result_middle3 = f3["result_middle_3"].value
        f4 = h5py.File(h4, 'r')
        self.result_middle4 = f4["result_middle_4"].value
        f5 = h5py.File(h5, 'r')
        self.result_middle5 = f5["result_middle_5"].value

        f = h5py.File(image_imput, 'r')
        self.result_input = f["result_input"].value

    def test_proposal_tf_layer(self):
        from quatization.lib.CustomLayer import ProposalTF

        input_shape = self.result_input[0].shape
        box_encodings = self.result_middle0
        class_predictions_with_background = self.result_middle_1
        rpn_box_predictor_features = self.result_middle2
        rpn_features_to_crop = self.result_middle3

        inputs0 = Input(shape=input_shape[1:])
        inputs1 = Input(shape=box_encodings.shape[1:])
        inputs2 = Input(shape=class_predictions_with_background.shape[1:])
        inputs3 = Input(shape=rpn_box_predictor_features.shape[1:])
        inputs4 = Input(shape=rpn_features_to_crop.shape[1:])

        inputs = [inputs0, inputs1, inputs2, inputs3, inputs4]
        # TODO 参数
        proposal = ProposalTF(scales=(0.25, 0.5, 1.0, 2.0), aspect_ratios=(0.5, 1.0, 2.0), anchor_stride=(16, 16),
                              max_proposals=100, nms_score_threshold=0.0, nms_iou_threshold=0.699999988079)(inputs)

        print(" ============= ProposalTF done =============")
        model = Model(inputs=inputs, outputs=proposal)
        data = [self.result_input[0], self.result_middle0, self.result_middle_1, self.result_middle2,
                self.result_middle3]
        proposaled = model.predict(data)
        print("keras proposaled result ===== :", proposaled.shape)
        print("keras proposaled result  0:", proposaled[0][0])
        # [0.01159281 0.4021027  0.6647914  0.9458419]

        # with h5py.File("proposaled.h5", "w") as f:
        #     f["proposaled"] = proposaled

    def test_roi_pooling_layer(self):
        from quatization.lib.CustomLayer import RoiPoolingTF
        rpn_features_to_crop = self.result_middle3

        with h5py.File("proposaled.h5", "r") as f:
            proposaled = f["proposaled"].value

        inputs0 = Input(shape=rpn_features_to_crop.shape[1:])
        inputs1 = Input(shape=proposaled.shape[1:])
        inputs = [inputs0, inputs1]
        # TODO
        roi_pooling = RoiPoolingTF(initial_crop_size=14, maxpool_kernel_size=2, maxpool_stride=2)(inputs)
        print(" ============= RoiPoolingTF done =============")
        model = Model(inputs=inputs, outputs=roi_pooling)
        data = [rpn_features_to_crop, proposaled]

        roi_pooled = model.predict(data)
        print("keras roi_pooled result ===== :", roi_pooled.shape)
        print("keras roi_pooled result  0:", roi_pooled[0, 0, :, :, 0])

    def test_proposal_caffe(self):
        from quatization.lib.CustomLayer import ProposalCaffe
        import time

        f = h5py.File("caffe_input0.h5", 'r')
        caffe_input0 = f["caffe_input0"].value
        f1 = h5py.File("caffe_input1.h5", 'r')
        caffe_input1 = f1["caffe_input1"].value


        inputs0 = Input(shape=caffe_input0.shape[1:])
        inputs1 = Input(shape=caffe_input1.shape[1:])
        inputs = [inputs0, inputs1]

        proposal_caffe = ProposalCaffe(feat_stride=16, scales=(8, 16, 32), pre_nms_topN=6000,
                   post_nms_topN=300, nms_thresh=0.5, min_size=16)(inputs)
        print(" ============= RoiPoolingTF done =============")
        model = Model(inputs=inputs, outputs=proposal_caffe)
        data = [caffe_input0, caffe_input1]

        proposaled_caffe = model.predict(data)

        print("keras proposaled_caffe result ===== :", proposaled_caffe.shape)

    def test_keras_RoiPoolingCaffe(self):
        from quatization.lib.CustomLayer import RoiPoolingCaffe
        import time

        f = h5py.File("ca_roi_input0.h5", 'r')
        ca_roi_input0 = f["ca_roi_input0"].value
        f1 = h5py.File("ca_roi_input1.h5", 'r')
        ca_roi_input1 = f1["ca_roi_input1"].value
        ca_roi_input1 = np.expand_dims(ca_roi_input1, axis=0)
        print("ca_roi_input1.shape:", ca_roi_input1.shape)

        start = time.time()

        inputs0 = Input(shape=ca_roi_input0.shape[1:])
        inputs1 = Input(shape=ca_roi_input1.shape[1:])
        inputs = [inputs0, inputs1]

        proposal_caffe = RoiPoolingCaffe(pooled_w=7, pooled_h=7, spatial_scale=0.0625)(inputs)
        print(" ============= RoiPoolingCaffe done =============")
        model = Model(inputs=inputs, outputs=proposal_caffe)
        data = [ca_roi_input0, ca_roi_input1]

        end0 = time.time()
        print("make graph time:", end0 - start)

        roi_pooled_caffe = model.predict(data)
        print("keras roi_pooled_caffe result ===== :", roi_pooled_caffe.shape)

        end1 = time.time()
        print("predict time:", end1 - end0)

        with h5py.File("keras_roi_test.h5", "w") as f2:
            f2["keras_roi"] = roi_pooled_caffe[0]

    def test_tf_op(self):

        outputs = tf.zeros(shape=[300, 7, 7, 512], dtype=tf.float32)
        ref = tf.Variable(outputs)
        indices = tf.constant([[0, 0, 0]], dtype=tf.int32)
        print("indices:", indices.shape)
        updates = tf.ones(shape=[1, 512], dtype=tf.float32)
        print("updates:", updates.shape)
        update = tf.scatter_nd_update(ref, indices, updates)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            updated = sess.run(update)
            print(updated.shape)


    def test_caffe_proposal_result(self):
        f = h5py.File("caffe_proposals.h5", 'r')
        caffe_proposals = f["caffe_proposals"].value

        f = h5py.File("keras_proposal.h5", 'r')
        keras_proposal = f["keras_proposal"].value

        diff = caffe_proposals - keras_proposal
        print("diff :", diff)
        print("max:", np.max(diff))

        #  [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  0.0000000e+00]]
        # max: 3.0517578e-05

    def test_op(self):

        with tf.Session() as sess:
            y = tf.ones(shape=512)
            input_ta = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            for time in range(7):
                input_ta = input_ta.write(time, y)  # 写入
            output = input_ta.stack()

            input_ta1 = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            for time in range(7):
                input_ta1 = input_ta1.write(time, output)  # 写入
            output1 = input_ta1.stack()

            input_ta2 = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
            for time in range(300):
                input_ta2 = input_ta2.write(time, output1)  # 写入
            output2 = input_ta2.stack()

            result = sess.run(output2)
            print(result.shape)


if __name__ == "__main__":
    unittest.main()
