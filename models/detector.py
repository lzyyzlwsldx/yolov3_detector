import os
import cv2
import numpy as np
import colorsys
import tensorflow as tf
from tensorflow.keras.models import load_model
from models.post_process import yolo_eval
from models.data_stream import get_model_input, get_anchors, get_classes, get_config
from scripts.utils import get_time
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
import pathlib


def mish(x):
    return x * tf.tanh(K.switch(tf.greater(x, 20), x, K.switch(tf.less(x, -20), tf.exp(x), K.log(tf.exp(x) + 1))))

class Detector:
    def __init__(self, ):
        pass


class Yolo(Detector):
    def __init__(self, model_cfg_dir):
        super().__init__()
        self.anchors = get_anchors(os.path.join(model_cfg_dir, 'anchors.txt'))
        self.class_names = get_classes(os.path.join(model_cfg_dir, 'classes.txt'))
        model_cfg = get_config(os.path.join(model_cfg_dir, 'config.ini'))
        self.input_shape = (int(model_cfg['height']), int(model_cfg['width']))
        self.hdf5_path = model_cfg['hdf5_path']
        weight_path = self.hdf5_path if os.path.exists(self.hdf5_path) else os.path.join(model_cfg_dir,
                                                                                         'backup/' + self.hdf5_path)
        self.model = self._generate(weight_path)

    def _generate(self, weight_path):
        assert weight_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        model = load_model(weight_path, custom_objects={'mish': mish}, compile=False)
        assert model.layers[-1].output_shape[-1] == len(self.anchors) / len(model.output) * (
                len(self.class_names) + 5), 'Mismatched between model and given anchors or classes'
        return model

    # def _generate_lite(self, weight_path):
    # """generate tensorflow-lite"""
    # assert weight_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
    # model = load_model(weight_path, custom_objects={'mish': mish}, compile=False)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    # tflite_model = converter.convert()
    # tflite_models_dir = pathlib.Path("./tmp/mnist_tflite_models/")
    # tflite_models_dir.mkdir(exist_ok=True, parents=True)
    # tflite_model_file = tflite_models_dir / "mnist_model.tflite"
    # tflite_model_file.write_bytes(tflite_model)
    # interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file), )
    # interpreter.allocate_tensors()
    # self.interpreter = interpreter

    # @get_time
    # def _detect_lite(self, inputs):
    # """detect by lite model"""
    #     # input_index = self.interpreter.get_input_details()[0]["index"]
    #     # output_index = self.interpreter.get_output_details()[0]["index"]
    #     start = time.time()
    #     self.interpreter.set_tensor(3, inputs)
    #     self.interpreter.invoke()
    #     outputs = [self.interpreter.get_tensor(2), self.interpreter.get_tensor(1), self.interpreter.get_tensor(0)]
    #     detections = yolo_eval(outputs, self.anchors, len(self.class_names),
    #                            max_boxes=20, score_threshold=.2, iou_threshold=.2)
    #     return detections

    @get_time
    @tf.function
    def _detect(self, inputs):
        outputs = self.model(inputs)
        detections = yolo_eval(outputs, self.anchors, len(self.class_names), max_boxes=20)
        return detections


    def perform_detect(self, image_path, show_image=False, verbose=False):
        if verbose:
            print('*' * 25 + ' Detecting ' + '*' * 25)
            print('Image path is: ', image_path)
        try:
            image, image_data = get_model_input(image_path, self.input_shape)
        except Exception as e:
            print(e)
            return [], ''
        detections, duration = self._detect(image_data)
        boxes, scores, classes = map(lambda x: x.numpy(), detections)
        detections = [[list(map(lambda x: int(round(x)), boxes[i])), float(scores[i]), int(classes[i])] for i in
                      range(len(scores))]
        if verbose:
            print('Time of detecting is ', duration)
            print('Detected {} boxes for {}'.format(len(detections), 'img'))
            print('Detections are ', detections)
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
    @tf.function
    def _detect(self, inputs):
        outputs = self.model(inputs)[::-1]
        detections = yolo_eval(outputs, self.anchors, len(self.class_names),
                               max_boxes=20, score_threshold=.2, iou_threshold=.2)
        return detections
