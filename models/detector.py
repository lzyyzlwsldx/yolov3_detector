import os
import cv2
import numpy as np
import colorsys
from tensorflow.keras.models import load_model
from models.post_process import yolo_eval
from models.data_stream import get_model_input, get_anchors, get_classes
from scripts.utils import get_time


class Detector:
    def __init__(self, ):
        pass


class Yolo(Detector):
    def __init__(self, model_cfg):
        super().__init__()
        self.__dict__.update(model_cfg)
        anchors_path, classes_path, model_path, = self.get_cfg_paths()
        self.anchors = get_anchors(anchors_path)
        self.class_names = get_classes(classes_path)
        self._generate(model_path)

    def get_cfg_paths(self):
        anchors_path = os.path.join(self.model_cfg_dir, 'anchors.txt')
        classes_path = os.path.join(self.model_cfg_dir, 'classes.txt')
        model_path = os.path.join(self.model_cfg_dir, 'backup/' + self.hdf5_name)
        return anchors_path, classes_path, model_path

    def _generate(self, model_path):
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model = load_model(model_path, compile=False)
        assert self.model.layers[-1].output_shape[-1] == len(self.anchors) / len(self.model.output) * (
                len(self.class_names) + 5), 'Mismatched between model and given anchors or classes'

    @get_time
    def _detect(self, inputs):
        outputs = self.model(inputs)
        boxes, scores, classes = yolo_eval(outputs, self.anchors, len(self.class_names),
                                           max_boxes=20)
        boxes, scores, classes = boxes.numpy(), scores.numpy(), classes.numpy()
        detections = [[list(map(lambda x: int(round(x)), boxes[i])), float(scores[i]), int(classes[i])] for i in
                      range(len(scores))]
        return detections

    def perform_detect(self, image_path, show_image=False, verbose=False):
        if verbose:
            print('*' * 25 + ' Detecting ' + '*' * 25)
            print('Image path is: ', image_path)
        try:
            image, image_data = get_model_input(image_path, self.input_size)
        except Exception as e:
            print(e)
            return [], ''
        detections, duration = self._detect(image_data)
        if verbose:
            print('Time of detecting is ', duration)
            print('Detected {} boxes for {}'.format(len(detections), 'img'))
            print('Detections are ', detections)
            # print('Length of filtered detections is: ', len(detections))
            # print('Filtered detections is :', detections)
        show_image and self._show_image(image, detections)
        return detections

    def _show_image(self, image, detections):
        try:
            thickness = max((image.shape[0] + image.shape[1]) // 1000, 2)
            hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
            np.random.seed(137)  # Fixed seed for consistent colors across runs.
            np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
            np.random.seed(None)  # Reset seed to default.
            for box, score, label in detections:
                print(label, ' ' * 5, score)
                left_top = tuple(box[:2][::-1])
                right_bottom = tuple(box[2:][::-1])
                image = cv2.rectangle(image, left_top, right_bottom, colors[int(label)], thickness)
                image = cv2.putText(image, self.class_names[int(label)], left_top, cv2.FONT_HERSHEY_PLAIN,
                                    thickness,
                                    color=colors[int(label)],
                                    thickness=thickness)
            cv2.imshow('result', image)
            cv2.waitKey()

        except Exception as e:
            print('Disable to show img.', str(e))
        return


class Csresnext(Yolo):
    @get_time
    def _detect(self, inputs):
        outputs = self.model(inputs)[::-1]
        boxes, scores, classes = yolo_eval(outputs, self.anchors, len(self.class_names),
                                           max_boxes=20)
        boxes, scores, classes = boxes.numpy(), scores.numpy(), classes.numpy()
        detections = [[list(map(lambda x: int(round(x)), boxes[i])), float(scores[i]), int(classes[i])] for i in
                      range(len(scores))]
        return detections
