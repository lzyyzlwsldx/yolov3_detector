import numpy as np
import cv2


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def letterbox_image(image, size):
    """
    Resize image with unchanged aspect ratio using padding.
    :param image: shape is (h, w, c) or (h, w)
    :param size: the size of dst.
    :return:
    """
    size = [size[0], size[1], image.shape[2]] if len(image.shape) > 2 else size
    ih, iw = image.shape[:2]
    dh, dw = size[:2]
    scale = min(dw / iw, dh / ih)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
    new_image = np.ones(size, dtype=np.uint8) * 128
    new_image[(dh - nh) // 2:(dh + nh) // 2, (dw - nw) // 2:(dw + nw) // 2] = image
    return new_image


def get_random_data(img, box, input_shape):
    """

    :param img:
    :param box:
    :param input_shape:
    :return:
    """
    if len(img.shape) > 2:
        input_shape = [input_shape[0], input_shape[1], img.shape[2]]
    ih, iw = img.shape[:2]
    dh, dw = input_shape[:2]
    scale = min(dw / iw, dh / ih)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(img, (nw, nh), cv2.INTER_CUBIC)
    new_image = np.ones(input_shape, dtype=np.uint8) * 128
    new_image[(dh - nh) // 2:(dh + nh) // 2, (dw - nw) // 2:(dw + nw) // 2] = image

    if len(box) > 0:
        np.random.shuffle(box)
        dx = (dw - nw) // 2
        dy = (dh - nh) // 2
        box[..., 2:4] = box[..., 2:4] * (nw / dw, nh / dh)
        box[..., :2] = (box[..., :2] * (nw, nh) + (dx, dy)) / (dw, dh)

    # for b in range(len(box)):
    #     x, y, w, h, c = box[b]
    #     box[b] = [max(0, int((x - w / 2) * dw)),
    #               max(0, int((y - h / 2) * dh)),
    #               min(dw, int((x + w / 2) * dw)),
    #               min(dh, int((y + h / 2) * dh)),
    #               int(c)]
    return new_image, box


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    boxes_wh = true_boxes[..., 2:4] * input_shape[::-1]

    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0
    # Discard zero rows.
    wh = boxes_wh[valid_mask]
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)

    for t, n in enumerate(best_anchor):
        for l in range(num_layers):
            if n in anchor_mask[l]:
                i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                k = anchor_mask[l].index(n)
                c = true_boxes[t, 4].astype('int32')
                y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                y_true[l][j, i, k, 4] = 1
                y_true[l][j, i, k, 5 + c] = 1
    for layer in range(num_layers):
        y_true_shape = y_true[layer].shape
        y_true[layer] = y_true[layer].reshape((y_true_shape[0], y_true_shape[1], -1))
    return y_true


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array([float(x) for x in anchors.split(',')]).reshape(-1, 2)
    return anchors


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = np.array([c.strip() for c in class_names])
    return class_names


def read_yolo_label(label_path):
    """

    :param label_path:
    :return: [x, y, w, h, c], normalized to 1.0
    """
    with open(label_path) as f:
        box_label = f.readlines()
    box_label = np.array([[v for v in b.split('\n')[0].split(' ')] for b in box_label], dtype=np.float16)
    box_label = box_label[..., [1, 2, 3, 4, 0]]
    return box_label


def val_mapper(set_paths, input_shape, anchors, num_classes):
    assert (np.array(input_shape) % 32 == 0).all(), 'Multiples of 32 required'
    for p in set_paths:
        img = cv2.imread(p)
        img, label = get_random_data(img, label, input_shape)
        img = img.astype(np.float32) / 255.
        y_true = preprocess_true_boxes(label, input_shape, anchors, num_classes)
        yield img, y_true[0], y_true[1], y_true[2]


def data_generator(set_paths, input_shape, anchors, num_classes, mode=''):
    assert (np.array(input_shape) % 32 == 0).all(), 'Multiples of 32 required'
    for p in set_paths:
        img = cv2.imread(p)
        # 读取yolo label
        label = read_yolo_label(p.split('.')[0] + '.txt')
        # 数据增强
        img, label = get_random_data(img, label, input_shape)
        img = img.astype(np.float32) / 255.
        y_true = preprocess_true_boxes(label, input_shape, anchors, num_classes)
        yield img, y_true[0], y_true[1], y_true[2]


def get_model_input(image_path, need_size):
    image = cv2.imread(image_path)
    if need_size != (0, 0):
        assert need_size[0] % 32 == 0, 'Multiples of 32 required'
        assert need_size[1] % 32 == 0, 'Multiples of 32 required'
        image = letterbox_image(image, need_size)
    else:
        new_image_size = (image.height - (image.height % 32), image.width - (image.width % 32))
        image = letterbox_image(image, new_image_size)
    image_data = np.expand_dims(image.astype(np.float32) / 255., 0)  # Add batch dimension.
    return image, image_data
