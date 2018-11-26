import unittest
import numpy as np
from platformx.plat_tensorflow.tools.processor.np_utils import shape_utils
import tensorflow as tf
import heapq
import h5py


def softmax_co(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def save_txt():
    out_boxes = np.ones(shape=(1917, 4))
    per_image_scores = np.ones(shape=(1917, 90))
    txt_concat = np.concatenate((out_boxes, per_image_scores), axis=1)

    ARRS = []
    f = open('ssd_mobilenet_nms.txt', 'w+')
    for i in range(txt_concat.shape[0]):
        jointsFrame = txt_concat[i]  # 每行
        ARRS.append(jointsFrame)
        for Ji in range(txt_concat.shape[1]):
            strNum = str(jointsFrame[Ji])
            f.write(strNum)
            f.write(',  ')
        f.write('\n')
    f.close()

def top_k(input, k=1):
    """Top k max pooling
    Args:
        input(ndarray): convolutional feature in heigh x width x channel format
        k(int): if k==1, it is equal to normal max pooling
    Returns:
        ndarray: k x (height x width)
    """
    input  = np.reshape(input, [-1, input.shape[-1]])
    input = np.sort(input, axis=0)[::-1, :][:k, :]
    return input

def get_least_numbers_big_data(alist, k):
    max_heap = []
    length = len(alist)
    if not alist or k <= 0 or k > length:
        return
    k = k - 1
    for ele in alist:
        ele = -ele
        if len(max_heap) <= k:
            heapq.heappush(max_heap, ele)
        else:
            heapq.heappushpop(max_heap, ele)

    return map(lambda x: -x, max_heap)



def qselect(A, k):
    if len(A) < k: return A
    pivot = A[-1]
    right = [pivot] + [x for x in A[:-1] if x >= pivot]
    rlen = len(right)
    if rlen == k:
        return right
    if rlen > k:
        return qselect(right, k)
    else:
        left = [x for x in A[:-1] if x < pivot]
        return qselect(left, k - rlen) + right


def topK_heapq(num_list, k):
    array = []
    for i in range(len(num_list)):
        if len(array) < k:
            heapq.heappush(array, num_list[i])
        else:
            array_min = array[0]
            if num_list[i] > array_min:
                heapq.heapreplace(array, num_list[i])
    topK = array

    return topK


class POSTALLMETHODTEST(unittest.TestCase):

    def setUp(self):
        self.np_array = np.ones(shape=(1, 300, 300, 3))

    def test_topK_heapq(self):

        field_to_sort = h5py.File('field_to_sort.h5', 'r')
        field_to_sort_np = field_to_sort["field_to_sort"].value

        ind = np.argpartition(field_to_sort_np, np.array(range(9000)))[::-1]

        print("ind shape:", ind.shape)
        print("ind:", ind)

    def test_h5py(self):

        test = np.ones(shape=(1, 9000))
        print(test.shape)
        import h5py
        f = h5py.File('test.h5', 'w')
        f["test"] = test


    def test_gather(self):
        import tensorflow as tf
        temp4 = tf.reshape(tf.range(0, 7668) + tf.constant(1, shape=[7668]), [1917, 4])
        temp5 = tf.gather(temp4, [0, 1], axis=0)  # indices是向量
        temp6 = tf.gather(temp4, 1, axis=1)  # indices是数值
        temp7 = tf.gather(temp4, [1, 3], axis=1)
        temp8 = tf.gather(temp4, [[0, 1], [1, 2]], axis=1)  # indices是多维的

        with tf.Session() as sess:
            result = sess.run(temp4)
            print("result:", result.shape)
            result1 = sess.run(temp5)
            print("result1:", result1.shape)
            result2 = sess.run(temp6)
            print("result2:", result2.shape)
            result3 = sess.run(temp7)
            print("result3:", result3.shape)

            # print(sess.run(temp6))
            # print(sess.run(temp7))
            # print(sess.run(temp8))

    def test_boxlist_filtered_diff(self):
        import h5py

        f_np = h5py.File('tf_proposal.h5', 'r')

        boxlist_filtered_np = f_np["tf_proposal"].value
        print("tf_proposal :", boxlist_filtered_np.shape)

        f_tf = h5py.File('caffe_proposal.h5', 'r')

        boxlist_filtered_tf = f_tf["caffe_proposal"].value
        print("caffe_proposal:", boxlist_filtered_tf.shape)

        print(" proposal diff:", boxlist_filtered_tf - boxlist_filtered_np)

        max_diff = np.max(np.abs(boxlist_filtered_tf - boxlist_filtered_np))
        print("max_diff:", max_diff)

    #     max_diff: 0.5111334831014741

    def test_boxlist_filtered_scores_diff(self):
        import h5py

        f_np = h5py.File('scores.h5', 'r')

        scores_np = f_np["scores"].value
        print("scores_np:", scores_np.shape)

        f_tf = h5py.File('boxlist_filtered_scores_tf.h5', 'r')

        boxlist_filtered_scores_tf = f_tf["boxlist_filtered_scores_tf"].value
        print("boxlist_filtered_scores_tf:", boxlist_filtered_scores_tf.shape)

        print(" scores diff:", boxlist_filtered_scores_tf - scores_np)

        max_diff = np.max(np.abs(boxlist_filtered_scores_tf - scores_np))
        print("max_diff:", max_diff)

        # scores diff: [-2.6885234e-04 -1.8966889e-03 -1.5004235e-04 ... -2.3458730e-02
        #   1.3514092e-02 -6.2564388e-05]
        # max_diff: 0.6964302

    def test_per_image_boxes_diff(self):
        import h5py

        f_np = h5py.File('per_image_boxes.h5', 'r')

        per_image_boxes_np = f_np["per_image_boxes"].value
        print("per_image_boxes_np:", per_image_boxes_np.shape)

        f_tf = h5py.File('per_image_boxes_tf.h5', 'r')

        per_image_boxes_tf = f_tf["per_image_boxes_tf"].value
        print("per_image_boxes_tf:", per_image_boxes_tf.shape)

        diff = per_image_boxes_tf - per_image_boxes_np

        print(" per_image_boxes diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    #     per_image_boxes_tf: (1917, 1, 4)
    #  per_image_boxes diff: [[[-0.01365474 -0.01097421  0.01154981 -0.00015201]]
    #
    #  [[-0.01083586  0.0387917   0.02279336 -0.01244337]]
    #
    #  [[ 0.00734656  0.01025141  0.00797784  0.01259335]]
    #
    #  ...
    #
    #  [[ 0.01399781 -0.01085355 -0.00952968  0.00447204]]
    #
    #  [[-0.01484327 -0.01331163  0.01603431 -0.02723793]]
    #
    #  [[ 0.00188727 -0.00149506  0.00445205  0.00081215]]]
    # max_diff: 0.5111334831014741

    def test_detection_boxes_diff(self):
        import h5py

        f_np = h5py.File('boxes.h5', 'r')
        detection_boxes_np = f_np["boxes"].value
        print("detection_boxes_np:", detection_boxes_np.shape)
        f_tf = h5py.File('result_final.h5', 'r')

        detection_boxes_tf = f_tf["result_final"].value
        print("detection_boxes_tf:", detection_boxes_tf.shape)

        diff = detection_boxes_tf - detection_boxes_np
        print(" detection_boxes diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_detection_scores_diff(self):
        import h5py

        f_np = h5py.File('detection_scores.h5', 'r')
        detection_boxes_np = f_np["detection_scores"].value
        print("np:", detection_boxes_np.shape)

        f_tf = h5py.File('detection_scores_tf.h5', 'r')

        detection_boxes_tf = f_tf["detection_scores_tf"].value
        print("tf:", detection_boxes_tf.shape)

        diff = detection_boxes_tf - detection_boxes_np

        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)


    def test_nn_softmax(self):

        rand = np.random.rand(1, 278, 2)
        print("rand:", rand.shape)
        # rand_h = tf.constant(rand, dtype=tf.float32)

        with tf.Session() as sess:
            out = tf.nn.softmax(rand)
            out = sess.run(out)
            # test = "Slice:0"
            # test_out = sess.run([test], feed_dict={rand_h: rand})
            # print("test_out:", test_out)
            print("out:", out.shape)

        rand1 = rand[0]

        print("rand1:", rand1.shape)

        with tf.Session() as sess:
            out_1 = tf.nn.softmax(rand1)
            out_1 = sess.run(out_1)

            print("out_1:", out_1.shape)

        print("diff:", out[0] - out_1)

        # my_soft = softmax(rand)
        co_soft = softmax_co(rand1)

        print("diff co_soft:", out[0] - co_soft)

        print("max co_soft:", np.max(out[0] - co_soft))


    def test_crop_and_resize_out_diff(self):
        import h5py

        f_np = h5py.File('fisrt_post_result.h5', 'r')
        detection_boxes_np = f_np["fisrt_post_result"].value
        print("np:", detection_boxes_np.shape)
        # print("np result:", detection_boxes_np[0][0])
        f_tf = h5py.File('crop_and_resize_out.h5', 'r')
        detection_boxes_tf = f_tf["crop_and_resize_out"].value[0]
        print("tf:", detection_boxes_tf.shape)
        # print("tf result:", detection_boxes_tf[0][0])
        diff = detection_boxes_tf - detection_boxes_np
        print(" diff:", diff)
        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)


    def test_diff(self):
        f_np = h5py.File('input_data.h5', 'r')
        detection_boxes_np = f_np["input_data"].value
        print("np:", detection_boxes_np.shape)

        # print("np result:", detection_boxes_np[0][0])

        f_tf = h5py.File('result_input.h5', 'r')

        detection_boxes_tf = f_tf["result_input"].value
        print("tf:", detection_boxes_tf.shape)
        # print("tf result:", detection_boxes_tf[0][0])
        diff = detection_boxes_tf - detection_boxes_np
        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)


    def test_proposal_boxes_normalized_diff(self):
        f_np = h5py.File('proposal_boxes_normalized.h5', 'r')
        detection_boxes_np = f_np["proposal_boxes_normalized"].value
        print("np:", detection_boxes_np.shape)

        # print("np result:", detection_boxes_np[0][0])

        f_tf = h5py.File('t_proposal_boxes_normalized.h5', 'r')

        detection_boxes_tf = f_tf["t_proposal_boxes_normalized"].value[0]
        print("tf:", detection_boxes_tf.shape)

        # print("tf result:", detection_boxes_tf[0][0])

        diff = detection_boxes_tf - detection_boxes_np

        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_boxlist_filtered_diff_1(self):
        import h5py

        f_np = h5py.File('proposal_boxes.h5', 'r')
        detection_boxes_np = f_np["proposal_boxes"].value
        print("np:", detection_boxes_np.shape)

        f_tf = h5py.File('t_proposal_boxes.h5', 'r')

        detection_boxes_tf = f_tf["t_proposal_boxes"].value
        print("tf:", detection_boxes_tf.shape)

        diff = detection_boxes_tf - detection_boxes_np

        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_selected_indices_diff_1(self):
        import h5py

        f_np = h5py.File('rpn_objectness_softmax_without_background.h5', 'r')
        detection_boxes_np = f_np["rpn_objectness_softmax_without_background"].value
        print("np:", detection_boxes_np.shape)

        f_tf = h5py.File('t_rpn_objectness_softmax.h5', 'r')

        detection_boxes_tf = f_tf["t_rpn_objectness_softmax"].value
        print("tf:", detection_boxes_tf.shape)

        diff = detection_boxes_tf - detection_boxes_np

        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_nms_result_scores_tf_diff(self):
        import h5py

        f_np = h5py.File('nms_result_scores_np.h5', 'r')
        detection_boxes_np = f_np["nms_result_scores_np"].value
        print("np:", detection_boxes_np.shape)

        f_tf = h5py.File('nms_result_scores_tf.h5', 'r')

        detection_boxes_tf = f_tf["nms_result_scores_tf"].value
        print("tf:", detection_boxes_tf.shape)

        diff = detection_boxes_tf - detection_boxes_np

        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_concated_np_diff(self):
        import h5py

        f_np = h5py.File('concated_np.h5', 'r')
        detection_boxes_np = f_np["concated_np"].value
        print("np:", detection_boxes_np.shape)

        f_tf = h5py.File('concated_tf.h5', 'r')

        detection_boxes_tf = f_tf["concated_tf"].value
        print("tf:", detection_boxes_tf.shape)

        diff = detection_boxes_tf - detection_boxes_np

        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_concatenated_field_diff(self):
        import h5py

        f_np = h5py.File('concatenated_field.h5', 'r')
        detection_boxes_np = f_np["concatenated_field"].value
        print("np:", detection_boxes_np.shape)

        f_tf = h5py.File('concatenated_field_tf.h5', 'r')

        detection_boxes_tf = f_tf["concatenated_field_tf"].value
        print("tf:", detection_boxes_tf.shape)

        diff = detection_boxes_tf - detection_boxes_np

        print(" diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_BoxEncodingPredictor_diff(self):
        import h5py

        f_np = h5py.File('tf_result_middle_0.h5', 'r')
        np_result = f_np["tf_result_middle_0"].value
        print("np_result:", np_result.shape)

        f_tf = h5py.File('BoxEncodingPredictor_2.h5', 'r')

        tf_result = f_tf["BoxEncodingPredictor_2"].value
        print("tf_result:", tf_result.shape)

        diff = tf_result - np_result

        print(" detection_boxes diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_sorted_indices_result_diff(self):
        import h5py

        f_np = h5py.File('sorted_indices_result.h5', 'r')
        np_result = f_np["sorted_indices_result"].value
        print("np_result:", np_result)

        f_tf = h5py.File('sorted_indices_tf.h5', 'r')
        tf_result = f_tf["sorted_indices_tf"].value
        print("tf_result:", tf_result)
        diff = tf_result - np_result
        print(" detection_boxes diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_field_to_sort_diff(self):
        import h5py

        f_np = h5py.File('field_to_sort.h5', 'r')
        np_result = f_np["field_to_sort"].value
        print("np_result:", np_result)

        f_tf = h5py.File('field_to_sort_tf.h5', 'r')
        tf_result = f_tf["field_to_sort_tf"].value
        print("tf_result:", tf_result)
        diff = tf_result - np_result
        print(" detection_boxes diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

        import tensorflow as tf

        with tf.Session() as sess:
            field_to_sort_holder = tf.placeholder(shape=list(tf_result.shape), dtype=tf.float32)
            num_boxes_holder = tf.constant(9000, dtype=tf.int32)

            with tf.device('/cpu:0'):
                top_ked = tf.nn.top_k(field_to_sort_holder, num_boxes_holder, sorted=True)

                top_ked_result = sess.run(top_ked, feed_dict={field_to_sort_holder: tf_result})

            print("top_ked_result:", top_ked_result)

    def test_input_data_diff(self):
        import h5py

        f_np = h5py.File('input_data.h5', 'r')
        np_result = f_np["input_data"].value
        print("np_result:", np_result.shape)

        f_tf = h5py.File('result_input.h5', 'r')

        tf_result = f_tf["result_input"].value
        print("tf_result:", tf_result.shape)

        diff = tf_result - np_result

        print("diff:", diff)

        max_diff = np.max(np.abs(diff))
        print("max_diff:", max_diff)

    def test_tf(self):
        grid_height = 19
        # Get a grid of box centers
        # y_centers = tf.to_float(tf.range(grid_height))
        # x_np = np.array([0.02631579, 0.07894737, 0.13157895, 0.18421053, 0.23684211, 0.28947368,
        #  0.34210526, 0.39473684, 0.44736842, 0.5,        0.55263158, 0.60526316,
        #  0.65789474, 0.71052632, 0.76315789, 0.81578947, 0.86842105, 0.92105263,
        #  0.97368421])

        width = np.array([56, 52, 96])

        print("width.ndim:", width.ndim)

        width_np = np.array(width.shape)
        print("width_np:", width_np)

        x_np_19 = np.ones(shape=(19, 19))

        final_x, final_y = meshgrid_np(width, x_np_19)

        print("final_x:", final_x.shape)

        print("final_y:", final_y.shape)

        width_p = tf.placeholder(shape=[3], dtype=tf.float32)
        x_np_p = tf.placeholder(shape=[19, 19], dtype=tf.float32)
        #
        out = meshgrid(width_p, x_np_p)

        sess = tf.Session()
        result = sess.run(out, feed_dict={x_np_p: x_np_19, width_p: width})
        #
        # shape = tf.shape(x_np_p)
        #
        # shape_np = sess.run(shape, feed_dict={x_np_p: x_np_19})
        # print("shape_np:", shape_np)

        print("len(result):", len(result))
        print("(result):", result[0].shape)

        # result_np = np.meshgrid(x_np, x_np)
        #
        # print("len(result_np):", len(result_np))
        # print("(result_np):", result_np[0])
        #
        diff = final_x - result[0]
        print("diff:", diff)

        # rank_y = tf.rank(y)
        #
        # rank_y_r = sess.run(rank_y)
        # print("rank_y_r ", rank_y_r)

        orig_shape = [3]

        orig_shape_b = orig_shape[0:0]
        print("orig_shape_b:", orig_shape_b)

        start_dim = 0
        num_dims = 2

        orig_shape_p = tf.constant([3], dtype=tf.int32)

        start_dim_p = tf.constant(0, dtype=tf.int32)
        num_dims_p = tf.constant(2, dtype=tf.int32)

        new_shape = expanded_shape(orig_shape_p, start_dim_p, num_dims_p)

        new_shape_r = sess.run(new_shape)

        print("new_shape_r:", new_shape_r)

        orig_shape_a = np.array([3], dtype=np.int32)

        expanded_r = expanded_shape_np(orig_shape_a, 0, 2)

        print("expanded_r:", expanded_r)

        # start_dim_p = tf.expand_dims(start_dim, 0)
        # start_dim_p = sess.run(start_dim_p)
        # print("start_dim_p:", start_dim_p)
        #
        # start_dim_np = np.expand_dims(start_dim, 0)
        #
        # print("start_dim_np:", start_dim_np)
        #
        # np.reshape(num_dims, [1])
        #
        # before = tf.slice(orig_shape_p, [0], start_dim_p)
        # before_r = sess.run(before)
        #
        # print("before_r:", before_r)

    def test_np_gather(self):
        a = np.array([i + 1 for i in range(7668)]).reshape(1917, 4)
        print("a:", a.shape)

        # res = gather(a, 1 , [[0],[8] , [19]]).flatten()
        # print("res:", res.shape)

        take_res = np.take(a, [3, 0, 1], axis=0)
        print("take_res:", take_res.shape)

        # res1 = gather(a, 0, 1)
        # print("res1:", res1)
        import tensorflow as tf
        temp4 = tf.reshape(tf.range(0, 7668) + tf.constant(1, shape=[7668]), [1917, 4])
        temp5 = tf.gather(temp4, [3, 0, 1], axis=0)  # indices是数值

        with tf.Session() as sess:
            result = sess.run(temp5)
            print("result:", result.shape)

        print("diff:", take_res - result)

    def test_unstack(self):
        import tensorflow as tf

        c = tf.constant([[1, 2, 3],

                         [4, 5, 6]])

        d = tf.unstack(c, axis=0)

        e = tf.unstack(c, axis=1)

        with tf.Session() as sess:
            print(sess.run(d))
            print(sess.run(e))

        x = np.array([[1, 2, 3], [4, 5, 6]])
        print("x.shape:", x.shape)
        x1 = np.split(x, 3, axis=1)
        print("x1:", x1)


def gather(a, dim, index):
    expanded_index = [index if dim == i else np.arange(a.shape[i]).reshape([-1 if i == j else 1 for j in range(a.ndim)])
                      for i in range(a.ndim)]
    return a[expanded_index]


def gather_numpy(arr, axis, index):
    """
    Gathers values along an axis specified by dim.

    :param axis: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:axis] + index.shape[axis + 1:]
    print("idx_xsection_shape:", idx_xsection_shape)
    self_xsection_shape = arr.shape[:axis] + arr.shape[axis + 1:]
    print("self_xsection_shape:", self_xsection_shape)

    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(axis) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(arr, 0, axis)
    index_swaped = np.swapaxes(index, 0, axis)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, axis)


def expanded_shape(orig_shape, start_dim, num_dims):
    with tf.name_scope('ExpandedShape'):
        start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
        print("tf start_dim:", start_dim.shape)
        before = tf.slice(orig_shape, [0], start_dim)

        add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
        after = tf.slice(orig_shape, start_dim, [-1])
        print("tf after:", after.shape)
        new_shape = tf.concat([before, add_shape, after], 0)
        return new_shape


def expanded_shape_np(orig_shape, start_dim, num_dims):
    start_dim = np.expand_dims(start_dim, 0)  # scalar to rank-1
    print("expand_dims start_dim:", start_dim)
    before = orig_shape[0:start_dim[0]]

    shape_p = np.reshape(num_dims, [1])
    print("shape_p:", shape_p)
    add_shape = np.ones(shape=shape_p, dtype=np.int32)
    # after = tf.slice(orig_shape, start_dim, [-1])
    after = orig_shape[start_dim[0]:]
    print("after:", after)
    new_shape = np.concatenate((before, add_shape, after), 0)
    return new_shape


def meshgrid(x, y):
    with tf.name_scope('Meshgrid'):
        print("x:", x.shape)
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
        print("x_exp_shape:", x_exp_shape.shape)
        y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

        xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
        ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
        new_shape = y.get_shape().concatenate(x.get_shape())

        xgrid.set_shape(new_shape)
        ygrid.set_shape(new_shape)

        print("xgrid:", xgrid.shape)
        # print("ygrid:", ygrid)
        return xgrid, ygrid


def meshgrid_np(x, y):
    x_exp_shape = expanded_shape_np(np.array(x.shape), 0, y.ndim)
    print("np x_exp_shape:", x_exp_shape.shape)
    y_exp_shape = expanded_shape_np(np.array(y.shape), y.ndim, x.ndim)
    print("np y_exp_shape:", y_exp_shape.shape)

    xgrid = np.tile(np.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = np.tile(np.reshape(y, y_exp_shape), x_exp_shape)

    print("y.shape:", y.shape)

    y_shape = np.array(y.shape, dtype=np.int32)
    x_shape = np.array(x.shape, dtype=np.int32)

    new_shape = np.concatenate((y_shape, x_shape))

    xgrid.reshape(new_shape)
    ygrid.reshape(new_shape)

    print("np xgrid:", xgrid.shape)
    return xgrid, ygrid


def compute_op(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    # exp_x = np.reshape(exp_x, (-1, exp_x.shape[-1]))
    softmax_x = exp_x / np.sum(exp_x)
    # softmax_x = np.reshape(softmax_x, x.shape)
    return softmax_x


def _flatten_outer_dims(logits):
    """Flattens logits' outer dimensions and keep its last dimension."""
    last_dim_size = logits.shape[-1]
    print("last_dim_size:", last_dim_size)
    output = np.reshape(logits, np.concatenate([[-1], [last_dim_size]], 0))
    return output


def softmax(logits, axis=-1):
    if axis == -1:
        logits = _flatten_outer_dims(logits)
        output = compute_op(logits)
        output = np.reshape(output, logits.shape)
        print("============= output")
        return output