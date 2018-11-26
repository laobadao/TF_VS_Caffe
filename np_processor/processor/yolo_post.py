# -*- coding: utf-8 -*-

import numpy as np
from platformx.plat_tensorflow.tools.processor.np_utils import ops
import random
import colorsys
import cv2
import config
import os
from PIL import Image, ImageDraw


def read_coco_labels():
    path = config.cfg.POSTPROCESSOR.PATH_TO_LABELS
    f = open(path)
    class_names = []
    for l in f.readlines():
        l = l.strip()  # 去掉回车'\n'
        class_names.append(l)
    f.close()
    # print("class_names:", class_names)
    return class_names


def load_coco_names():
    file_name = config.cfg.POSTPROCESSOR.PATH_TO_LABELS
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name

    # print("names:", names)
    return names

class_names = read_coco_labels()


def decode(model_output):
    """
    yolov2 decode
    :param model_output:  darknet19 网络输出的特征图
    :param output_sizes:  darknet19网络输出的特征图大小，默认是 13*13(默认输入416*416，下采样32)
    :param num_class:
    :param anchors:
    :return:
    """
    #  output_sizes=(13, 13)
    output_sizes = config.cfg.PREPROCESS.HEIGHT // 32, config.cfg.PREPROCESS.WIDTH // 32  # 特征图尺寸是图片下采样 32 倍
    num_class = config.cfg.POSTPROCESSOR.NUM_CLASSES

    # if num_class is None:
    #     num_class = len(class_names)

    anchors = config.cfg.POSTPROCESSOR.ANCHORS
    anchors = np.array(anchors)
    H, W = output_sizes
    num_anchors = len(anchors)  # 这里的 anchor 是在configs文件中设置的
    print("num_anchors:", num_anchors)
    # anchors = tf.constant(anchors, dtype=tf.float32)  # 将传入的 anchors 转变成 tf 格式的常量列表

    # 13*13*num_anchors*(num_class+5)，第一个维度自适应 batchsize
    print("model_output:", model_output.shape)
    detection_result = np.reshape(model_output, [-1, H * W, num_anchors, num_class + 5])

    print("detection_result:", detection_result.shape)

    # darknet19 网络输出转化——偏移量、置信度、类别概率
    xy_offset = ops.sigmoid(detection_result[:, :, :, 0:2])  # 中心坐标相对于该 cell 左上角的偏移量，sigmoid 函数归一化到0-1
    wh_offset = np.exp(detection_result[:, :, :, 2:4])  # 相对于 anchor 的 wh 比例，通过 e 指数解码
    obj_probs = ops.sigmoid(detection_result[:, :, :, 4])  # 置信度，sigmoid 函数归一化到 0-1
    class_probs = ops.softmax(detection_result[:, :, :, 5:])  # 网络回归的是'得分',用 softmax 转变成类别概率
    # 构建特征图每个cell的左上角的xy坐标
    height_index = range(H)  # range(0,13)
    width_index = range(W)  # range(0,13)
    # 变成x_cell=[[0,1,...,12],...,[0,1,...,12]]和y_cell=[[0,0,...,0],[1,...,1]...,[12,...,12]]
    x_cell, y_cell = np.meshgrid(height_index, width_index)
    x_cell = np.reshape(x_cell, [1, -1, 1])  # 和上面[H*W,num_anchors,num_class+5]对应
    print("x_cell:", x_cell.shape)
    y_cell = np.reshape(y_cell, [1, -1, 1])
    print("y_cell:", y_cell.shape)

    # decode
    bbox_x = (x_cell + xy_offset[:, :, :, 0]) / W
    bbox_y = (y_cell + xy_offset[:, :, :, 1]) / H
    bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) / W
    bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) / H
    # 中心坐标+宽高 box(x,y,w,h) -> xmin=x-w/2 -> 左上+右下box(xmin,ymin,xmax,ymax)
    bboxes = np.stack([bbox_x - bbox_w / 2, bbox_y - bbox_h / 2,
                       bbox_x + bbox_w / 2, bbox_y + bbox_h / 2], axis=3)

    bboxes = np.reshape(bboxes, bboxes.shape)
    print("yolov2 decode bboxes:", bboxes.shape)
    print("yolov2 decode obj_probs:", obj_probs.shape)
    print("yolov2 decode class_probs:", class_probs.shape)

    return bboxes, obj_probs, class_probs

def _get_image_path():
    img_dir = config.cfg.PREPROCESS.IMG_LIST
    file_list = os.listdir(img_dir)
    image_path = os.path.join(img_dir, file_list[0])
    return image_path


def _get_image():
    image_path = _get_image_path()
    image = cv2.imread(image_path)
    return image


def _get_image_PIL():
    image_path = _get_image_path()
    image = Image.open(image_path)
    return image


# 【2】筛选解码后的回归边界框——NMS(post process后期处理)
def postprocess(bboxes, obj_probs, class_probs):

    threshold = config.cfg.POSTPROCESSOR.SCORE_THRESHOLD

    image = _get_image()
    image_shape = image.shape[:2]
    print("image_shape ", image_shape)
    if image_shape is None:
        image_shape = (416, 416)

    # bboxes 表示为：图片中有多少 box 就多少行；4 列分别是 box(xmin,ymin,xmax,ymax)
    bboxes = np.reshape(bboxes, [-1, 4])
    # 将所有 box 还原成图片中真实的位置
    bboxes[:, 0:1] *= float(image_shape[1])  # xmin*width
    bboxes[:, 1:2] *= float(image_shape[0])  # ymin*height
    bboxes[:, 2:3] *= float(image_shape[1])  # xmax*width
    bboxes[:, 3:4] *= float(image_shape[0])  # ymax*height
    bboxes = bboxes.astype(np.int32)

    # (1)cut the box:将边界框超出整张图片(0,0)—(415,415)的部分cut掉
    bbox_min_max = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
    bboxes = bboxes_cut(bbox_min_max, bboxes)

    # ※※※置信度*max类别概率=类别置信度scores※※※
    obj_probs = np.reshape(obj_probs, [-1])
    class_probs = np.reshape(class_probs, [len(obj_probs), -1])
    class_max_index = np.argmax(class_probs, axis=1)  # 得到max类别概率对应的维度
    class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
    scores = obj_probs * class_probs

    # ※※※类别置信度scores>threshold的边界框bboxes留下※※※
    keep_index = scores > threshold
    class_max_index = class_max_index[keep_index]
    scores = scores[keep_index]
    bboxes = bboxes[keep_index]

    # (2)排序 top_k (默认为400)
    class_max_index, scores, bboxes = bboxes_sort(class_max_index, scores, bboxes)
    # ※※※(3)NMS※※※
    class_max_index, scores, bboxes = bboxes_nms(class_max_index, scores, bboxes)

    draw_detection(image, bboxes, scores, class_max_index, class_names)

    return bboxes, scores, class_max_index


# 【3】绘制筛选后的边界框
def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / float(len(labels)), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if scores[i] < thr:
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 300)
        cv2.rectangle(imgcv, (box[0], box[1]), (box[2], box[3]), colors[cls_indx], thick)
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)
        # cv2.rectangle(imgcv, (box[0], box[1]-20), ((box[0]+box[2])//3+120, box[1]-8), (125, 125, 125), -1)  # puttext函数的背景
        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, (255, 255, 255), thick // 3)

    cv2.imwrite("yolov2_detect_result.jpg", imgcv)
    print('YOLO_v2 detection has done!')


######################## 对应【2】:筛选解码后的回归边界框 #########################################

# (1)cut the box:将边界框超出整张图片(0,0)—(415,415)的部分cut掉
def bboxes_cut(bbox_min_max, bboxes):
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_min_max = np.transpose(bbox_min_max)
    # cut the box
    bboxes[0] = np.maximum(bboxes[0], bbox_min_max[0])  # xmin
    bboxes[1] = np.maximum(bboxes[1], bbox_min_max[1])  # ymin
    bboxes[2] = np.minimum(bboxes[2], bbox_min_max[2])  # xmax
    bboxes[3] = np.minimum(bboxes[3], bbox_min_max[3])  # ymax
    bboxes = np.transpose(bboxes)
    return bboxes


# (2)按类别置信度 scores 降序，对边界框进行排序并仅保留 top_k
def bboxes_sort(classes, scores, bboxes, top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes, scores, bboxes


# (3)计算IOU+NMS
# 计算两个box的IOU
def bboxes_iou(bboxes1, bboxes2):
    print("bboxes1 before:", bboxes1.shape)
    print("bboxes2 before:", bboxes2.shape)
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    print("bboxes1 after:", bboxes1.shape)
    print("bboxes2 after:", bboxes2.shape)

    b1_x0, b1_y0, b1_x1, b1_y1 = bboxes1[0], bboxes1[1], bboxes1[2], bboxes1[3]
    b2_x0, b2_y0, b2_x1, b2_y1 = bboxes2[0], bboxes2[1], bboxes2[2], bboxes2[3]

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(b1_x0, b2_x0)
    int_xmin = np.maximum(b1_y0, b2_y0)
    int_ymax = np.minimum(b1_x1, b2_x1)
    int_xmax = np.minimum(b1_y1, b2_y1)

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    # 计算IOU
    int_vol = int_h * int_w  # 交集面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积
    IOU = int_vol / (vol1 + vol2 - int_vol)  # IOU=交集/并集
    return IOU


def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, y1] (以整个 feartrue map 左上角为中心点)
    :param box2: same as box1
    :return: IoU
    """

    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

# NMS，或者用 tf.image.non_max_suppression(boxes, scores,self.max_output_size, self.iou_threshold)
def bboxes_nms(classes, scores, bboxes):

    iou_threshold = config.cfg.POSTPROCESSOR.IOU_THRESHOLD

    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size - 1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i + 1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < iou_threshold, classes[(i + 1):] != classes[i])
            keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]


# ================================================ yolov3 post ================================================


def yolov3_decode_and_nms(model_output, img_size):

    num_classes = config.cfg.POSTPROCESSOR.NUM_CLASSES
    _ANCHORS = config.cfg.POSTPROCESSOR.ANCHORS

    out_shape = {}
    for i in range(len(model_output)):
        print("yolov3 keras result:", model_output[i].shape)
        out_shape[i] = model_output[i].shape[1]

    print("out_shape 1 :", out_shape)
    sorted_out_shape = sorted(out_shape.items(), key=lambda x: x[1])
    print("sorted_out_shape 2 :", sorted_out_shape)

    detect_1 = _detection_layer(model_output[sorted_out_shape[0][0]], num_classes, _ANCHORS[6:9], img_size)
    detect_2 = _detection_layer(model_output[sorted_out_shape[1][0]], num_classes, _ANCHORS[3:6], img_size)
    detect_3 = _detection_layer(model_output[sorted_out_shape[2][0]], num_classes, _ANCHORS[0:3], img_size)
    detections = np.concatenate([detect_1, detect_2, detect_3], axis=1)
    detections = detections_boxes(detections)

    result = non_max_suppression(detections)
    return result


def _detection_layer(predictions, num_classes, anchors, img_size):
    num_anchors = len(anchors)
    grid_size = predictions.shape[1:3]
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes
    predictions = np.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
    box_centers, box_sizes, confidence, classes = np.split(predictions, [2, 4, 5], axis=-1)

    box_centers = ops.sigmoid(box_centers)
    confidence = ops.sigmoid(confidence)

    grid_x = range(grid_size[0])
    grid_y = range(grid_size[1])
    a, b = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(a, (-1, 1))
    y_offset = np.reshape(b, (-1, 1))

    x_y_offset = np.concatenate([x_offset, y_offset], axis=-1)
    x_y_offset = np.reshape(np.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = np.tile(anchors, [dim, 1])
    box_sizes = np.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride
    detections = np.concatenate([box_centers, box_sizes, confidence], axis=-1)
    classes = ops.sigmoid(classes)
    predictions = np.concatenate([detections, classes], axis=-1)
    return predictions


def detections_boxes(detections):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.

    :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
    :return: converted detections of same shape as input
    """
    center_x, center_y, width, height, attrs = np.split(detections, [1, 2, 3, 4], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = np.concatenate([x0, y0, x1, y1], axis=-1)
    detections = np.concatenate([boxes, attrs], axis=-1)
    return detections


def non_max_suppression(predictions_with_boxes):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """

    confidence_threshold = config.cfg.POSTPROCESSOR.SCORE_THRESHOLD
    iou_threshold = config.cfg.POSTPROCESSOR.IOU_THRESHOLD

    print("predictions_with_boxes:", predictions_with_boxes.shape)
    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))
        bboxes = []
        scores = []
        class_max_index = []
        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                bboxes.append(box)
                scores.append(score)
                class_max_index.append(cls)
                cls_boxes = cls_boxes[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    image = _get_image_PIL()
    classes = load_coco_names()
    draw_boxes(result, image, classes, (416, 416))
    image.save("yolov3_result.jpg")

    bboxes = np.array(bboxes)
    scores = np.array(scores)
    class_max_index = np.array(class_max_index)
    print("bboxes:", bboxes)
    print("scores:", scores)
    print("class_max_index:", class_max_index)
    new_result = {}
    new_result['detection_boxes'] = bboxes
    new_result['detection_scores'] = scores
    new_result['detection_classes'] = class_max_index
    return new_result


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            print("score:", score)
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            print("draw_boxes box:", box)
            draw.rectangle(box, outline=color)
            print("cls_names[cls]:", cls_names[cls])
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)



