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
