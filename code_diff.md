
# Tensorflow code( numpy 实现)

1. 先 clip 2. 后 decode

## clip_to_window()

```
# clip_window:', array([   0,    0,  600, 1002]))
clip_window = np.array([0, 0, image_shape[1], image_shape[2]])

# 'before clip anchors:', array([-45.254834, -22.627417,  45.254834,  22.627417])
anchors = clip_to_window(anchors, clip_window)
#'after clip anchors:', array([ 0.      ,  0.      , 45.254834, 22.627417])

```

```
#  对 anchors 进行 clip 裁剪操作
def clip_to_window(boxlist, window, filter_nonoverlapping=True):

    # [   0,    0,  600, 1002] resize 后的尺寸 anchors 返回的时候就是按照 y_min, x_min, y_max, x_max
    y_min, x_min, y_max, x_max = np.split(boxlist, 4, axis=1)

    win_y_min, win_x_min, win_y_max, win_x_max = np.split(window, 4)    
    # win_y_min = 0, win_x_min = 0, win_y_max = 600 , win_x_max = 1002

    # [-45.254834, -22.627417,  45.254834,  22.627417]
    y_min_clipped = np.maximum(np.minimum(y_min, win_y_max), win_y_min)
    # max(min(0, 45), -45) = 0
    y_max_clipped = np.maximum(np.minimum(y_max, win_y_max), win_y_min)
    # max(min(1002, 45), -45) = 45
    x_min_clipped = np.maximum(np.minimum(x_min, win_x_max), win_x_min)
    # max(min(0, 22), -22) = 0
    x_max_clipped = np.maximum(np.minimum(x_max, win_x_max), win_x_min)
    # max(min(1002, 22), -22) = 22

    # [ 0.      ,  0.      , 45.254834, 22.627417]

    clipped = np.concatenate([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], 1)

    if filter_nonoverlapping:
        areas = _area(clipped) # 计算框的面积
        greater = np.greater(areas, 0.0)
        where = np.where(greater)
        # 'where:', (array([    0,     1,     2, ..., 28725, 28726, 28727]),))
        reshaped = np.reshape(where, [-1])
        nonzero_area_indices = reshaped
        clipped = _gather(clipped, nonzero_area_indices)

    return clipped

def _area(boxlist):
    """Computes area of boxes.

    Args:
      boxlist: BoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing box areas.
    """

    y_min, x_min, y_max, x_max = np.split(
        boxlist, 4, axis=1)

    return np.squeeze((y_max - y_min) * (x_max - x_min), [1])

def _gather(boxlist, indices):
    if indices.ndim != 1:
        raise ValueError('indices should have rank 1')
    if indices.dtype != np.int32 and indices.dtype != np.int64:
        raise ValueError('indices should be an int32 / int64 tensor')

    gathered_result = np.take(boxlist, indices, axis=0)

    return gathered_result
```

## decode( )

```
def bbox_transform_inv_tf(anchors, rel_codes):
    _scale_factors = [10.0, 10.0, 5.0, 5.0] # 比例因子
    print("============== bbox_transform_inv_tf _decode =============")
    """Decode relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    # rel_codes: (28728, 4)
    print("rel_codes:", rel_codes.shape)    
    # 获取 box 的中心点宽高，要与之前生成 anchors 的计算保持一致
    ycenter_a, xcenter_a, ha, wa = _get_center_coordinates_and_sizes(anchors)

    tras_rel_codes = np.transpose(rel_codes)
    # rel_codes[0]: [ 0.8462524 -0.1174521 -2.104301  -1.7837868]
    print("tras_rel_codes:", tras_rel_codes.shape)
    # tras_rel_codes: (4, 28728)
    # 第一段网络前向计算得到的 box 偏移量坐标存储的顺序，就是 ty, tx, th, tw ，这也是差异中的一点
    ty, tx, th, tw = np.split(tras_rel_codes, 4)
    ty = ty.flatten()
    tx = tx.flatten()
    th = th.flatten()
    tw = tw.flatten()
    # ty: 0.8462524
    # tx: -0.1174521
    # th: -2.104301
    # tw: -1.7837868
    # 差异二 都先除以 一个比例因子
    ty /= _scale_factors[0]
    tx /= _scale_factors[1]
    th /= _scale_factors[2]
    tw /= _scale_factors[3]

    # ty: 0.08462524
    # tx: -0.01174521
    # th: -0.4208602
    # tw: -0.35675734

    # tw 偏移量除以 比例因子后的值 再求指数，再乘以 anchor 的 wa
    w = np.exp(tw) * wa
    h = np.exp(th) * ha

    # w = exp(-0.35675734) * wa
    # w = 15.837887436758514  = 0.6999423251332363 *  22.627416997969522    
    # h 29.708977689240264 , ha = 45.25483399593904  
    # 中心点 偏移量除以 比例因子后的值 在加上 anchor ycenter_a
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    # ycenter_a = 22.62741699796952 xcenter_a = 11.313708498984761
    # ycenter = 0.08462524 * 45.25483399593904 + 22.62741699796952
    # ycenter = 26.457118036244527 ,
    # xcenter 11.047944738983285 ,

    # 再将上面算出的 宽高 中心点，转换成 坐标 为最终 decode 后的结果
    # h, w 是解码后的宽高  h = 29.708977689240264  w = 15.837887436758514
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.

    # ymin = 26.457118036244527 - 29.708977689240264/2 = 11.602629191624395

    decode_proposal = np.transpose(np.stack([ymin, xmin, ymax, xmax]))
    print("decode_proposal:", decode_proposal.shape)
    decode_proposal = decode_proposal.astype('float32', copy=False)
    # decode_proposal[0] =  [11.60262919  3.12900102 41.31160688 18.96688846]
    # 先 decode 再 clip [ 0.     ,  0.     , 37.36838, 15.30636] 差异很大，不可以随表换顺序
    return decode_proposal

def _get_center_coordinates_and_sizes(box_corners):
    """Computes the center coordinates, height and width of the boxes.

    Args:
      scope: name scope of the function.

    Returns:
      a list of 4 1-D tensors [ycenter, xcenter, height, width].
    """
    ymin, xmin, ymax, xmax = np.stack(np.transpose(box_corners))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]

```

## batch_nms()

```

def apply_nms(self, boxes, confs, nms_threshold, eta, top_k, max_selection_size=None):
     """
     Greedily selects a subset of bounding boxes in descending order of score.

     :param boxes: boxes
     :param confs: scores
     :param nms_threshold:
     :param eta:
     :param top_k:
     :param max_selection_size: for tensorflow ssd model ,select max_selection_size num from all bounding boxes
     :return:
     """
     idx_array = []

     adaptive_threshold = nms_threshold
     max_idx_list = self._get_max_idx(confs)
     for i in range(len(max_idx_list)):
         if max_selection_size is None:
             if i >= top_k and top_k > 0:
                 break
         else:
             if len(idx_array) == max_selection_size:
                 break

         keep = True
         idx = max_idx_list[i]
         for r_i in idx_array:
             if keep == False:
                 break
             overlap = self._iou(boxes[idx], boxes[r_i])
             keep = (overlap <= adaptive_threshold)

         if keep:
             idx_array.append(idx)

         if keep and eta < 1 and adaptive_threshold > 0.5:
             adaptive_threshold = adaptive_threshold * eta

     return idx_array

 def _get_max_idx(self, data_list):
     def _take_second(elem):
         return elem[1] if len(elem) > 0 else 0

     sort_data = []
     for i in range(len(data_list)):
         sort_data.append((i, data_list[i]))
     sort_data.sort(key=_take_second, reverse=True)

     sort_idx = []
     for i in range(len(sort_data)):
         sort_idx.append(sort_data[i][0])
     return sort_idx

 def _iou(self, box1, box2):
     # compute intersection
     inter_upleft = np.maximum(box1[:2], box2[:2])
     inter_botright = np.minimum(box1[2:], box2[2:])
     inter_wh = inter_botright - inter_upleft
     inter_wh = np.maximum(inter_wh, 0)
     inter = inter_wh[0] * inter_wh[1]
     # compute union
     area_pred = (box2[2] - box2[0]) * (box2[3] - box2[1])
     area_gt = (box1[2] - box1[0]) * (box1[3] - box1[1])
     union = area_pred + area_gt - inter
     # compute iou
     iou = inter / union
     return iou

```



# Caffe code

1. 先 decode 2. 再 clip

## bbox_transform_inv( )

```
def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    #   w = anchor[2] - anchor[0] + 1
    #   h = anchor[3] - anchor[1] + 1
    #   x_ctr = anchor[0] + 0.5 * (w - 1)
    #   y_ctr = anchor[1] + 0.5 * (h - 1)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    # print("dx {}, dy {}, dw {}, dh {} ".format(dx[0], dy[0], dw[0], dh[0]))

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # 差异，这个直接取出来做 exp
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    print("===== bbox_transform_inv pred_boxes:", pred_boxes.shape)
    # [-8.414413, 44.588097,  2.803227, 52.65213 ]
    return pred_boxes
    ```

## clip_boxes()

  ```
  def clip_boxes(boxes, im_shape):
    """
    对 decode 后的 proposal 进行 clip
    Clip boxes to image boundaries.
    """

    # x1 >= 0 im_shape[1] = 63  im_shape[0] = 38
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    # y_min_clipped = np.maximum(np.minimum(y_min, win_y_max), win_y_min)
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)

    return boxes
    # tf  [11.60262919  3.12900102 41.31160688 18.96688846] 已经没有可比性了
    # caffe array([ 0.      , 37.      ,  2.803227, 37.      ], dtype=float32))
  ```

## nms()

```

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
```
