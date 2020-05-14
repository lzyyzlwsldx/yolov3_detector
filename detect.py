import os
from absl import (app, flags)
from models.detector import Yolo, Csresnext

FLAGS = flags.FLAGS
# 模型配置目录，包括anchors，classes，backup等，目录下名字需保持一致。
flags.DEFINE_string('model_cfg_dir', None, 'The model_cfg dir for detecting.')
flags.DEFINE_string('image_dir', None, 'The image dir for detecting.')
USE_CPU = False

if USE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def model_init(model_cfg_dir):
    detector = Csresnext(model_cfg_dir)
    return detector


def main(_args):
    detector = model_init(FLAGS.model_cfg_dir)
    names = [n for n in os.listdir(FLAGS.image_dir) if n[-3:] in ['jpg', 'peg', 'bmp', 'png']]
    for n in names:
        detections = detector.perform_detect(FLAGS.image_dir + n, show_image=True, verbose=True)


if __name__ == '__main__':
    app.run(main)
