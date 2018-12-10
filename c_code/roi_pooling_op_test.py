import tensorflow as tf
import numpy as np
import roi_pooling_op
import h5py
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# array = np.random.rand(1, 24, 32, 3)
# data = tf.convert_to_tensor(array, dtype=tf.float32)
# rois = tf.convert_to_tensor([[10, 10, 20, 20],
#                              [10, 10, 20, 20],
#                              [10, 10, 20, 20],
#                              [30, 30, 40, 40]], dtype=tf.float32)
#
# W = weight_variable([3, 3, 3, 512])
# h = conv2d(data, W)

f = h5py.File("ca_roi_input0.h5", 'r')
ca_roi_input0 = f["ca_roi_input0"].value
f1 = h5py.File("ca_roi_input1.h5", 'r')
ca_roi_input1 = f1["ca_roi_input1"].value

h = tf.convert_to_tensor(ca_roi_input0, dtype=tf.float32)

roi_index = np.zeros(shape=(300, 1))

ca_roi_input1 = np.concatenate((roi_index, ca_roi_input1), axis=1)
print("========= new ca_roi_input1 ", ca_roi_input1.shape)

rois = tf.convert_to_tensor(ca_roi_input1, dtype=tf.float32)

print("============== ca_roi_input1.shape:", ca_roi_input1.shape)
print("============== ca_roi_input0.shape:", ca_roi_input0.shape)
[y, argmax] = roi_pooling_op.roi_pool(h, rois, 7, 7, 0.0625)
# y_data = tf.convert_to_tensor(np.ones((2, 6, 6, 1)), dtype=tf.float32)

# Minimize the mean squared errors.
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
result = sess.run(y)
print("result:", result.shape)
with h5py.File("keras_roi_c.h5", "w") as f2:
    f2["keras_roi_c"] = result
sess.close()

