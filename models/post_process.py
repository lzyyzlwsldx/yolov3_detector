import tensorflow as tf


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)

    # Reshape to (batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.cast(tf.reshape(anchors, [1, 1, 1, num_anchors, 2]), feats.dtype)
    grid_shape = tf.shape(feats)[1:3]  # height, width
    # 生成坐标处的网格值shape=（...,2)，注意网格值和索引值不同，是相反的。
    grid_x, grid_y = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
    new_grid_shape = (grid_shape[0], grid_shape[1], 1, 1)
    grid = tf.concat((tf.reshape(grid_x, new_grid_shape), tf.reshape(grid_y, new_grid_shape)), axis=-1)
    grid = tf.cast(grid, feats.dtype)
    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[::-1], feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], feats.dtype)
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.sigmoid(feats[..., 5:])
    return [grid, feats, box_xy, box_wh] if calc_loss else [box_xy, box_wh, box_confidence, box_class_probs]


def yolo_correct_boxes(box_xy, box_wh, input_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = tf.cast(input_shape, box_yx.dtype)
    boxes_min = box_yx - (box_hw / 2.)
    boxes_max = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        boxes_min[..., 0:1],  # y_min
        boxes_min[..., 1:2],  # x_min
        boxes_max[..., 0:1],  # y_max
        boxes_max[..., 1:2]  # x_max
    ], axis=-1)
    boxes *= tf.concat([input_shape, input_shape], axis=0)
    return boxes


def yolo_boxes_and_scores(one_layer_features, anchors, num_classes, input_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(one_layer_features, anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape)
    boxes = tf.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


# @tf.function
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """
    Evaluate YOLO model on given input and return filtered boxes.
    :param yolo_outputs: 模型的输出结果，这里shape=(3,)，每一层的具体数量根据图像卷积到最后的结果而定。
    :param anchors: 具体的anchor值，shape=(-1, 2)
    :param num_classes: 类别数量
    :param max_boxes:
    :param score_threshold:
    :param iou_threshold:
    :return:
    """
    num_layers = len(yolo_outputs)
    # 根据config里的顺序
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # 获取原始输入形状
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32
    boxes, box_scores = [], []
    for layer in range(num_layers):
        # read anchors of each feats in anchors.txt.
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[layer],
                                                    tf.gather(anchors, anchor_mask[layer]),
                                                    num_classes,
                                                    input_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)
    mask = box_scores >= score_threshold
    max_boxes_tensor = tf.constant(max_boxes, dtype='int32')
    boxes_, scores_, classes_ = [], [], []

    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(class_boxes,
                                                 class_box_scores,
                                                 max_boxes_tensor,
                                                 iou_threshold=iou_threshold)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)

    return boxes_, scores_, classes_
