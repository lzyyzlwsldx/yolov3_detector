import os
from absl import (app, flags)
from models.detector import Yolo, Csresnext

FLAGS = flags.FLAGS
flags.DEFINE_string('image_dir', None, 'The image dir for detecting.')
USE_CPU = False

if USE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def model_init():
    # 模型配置目录，包括anchors，classes，backup等，目录下名字需保持一致。
    model_cfg_dir = 'model_cfgs/chip_cfg'
    # 仅需要输入模型名，保存在配置目录的backup目录下
    hdf5_name = 'keras.h5'
    # 检测的图片尺寸
    input_size = (896, 1280)
    model_cfg = {'model_cfg_dir': model_cfg_dir,
                 'hdf5_name': hdf5_name,
                 'input_size': input_size,
                 }
    # 选择模型初始化
    detector = Csresnext(model_cfg)
    return detector


def main(_args):
    detector = model_init()
    names = [n for n in os.listdir(FLAGS.image_dir) if n[-3:] in ['jpg', 'peg', 'bmp', 'png']]
    for n in names:
        detections = detector.perform_detect(FLAGS.image_dir + n, show_image=False, verbose=True)


if __name__ == '__main__':
    app.run(main)
