import os
from models.detector import Yolo, Csresnext

USE_CPU = False

if USE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def model_init():
    model_cfg_dir = 'model_cfgs/chip_cfg'
    hdf5_name = 'keras.h5'
    input_size = (1312, 1312)
    model_cfg = {'model_cfg_dir': model_cfg_dir,
                 'hdf5_name': hdf5_name,
                 'input_size': input_size,
                 }

    return Csresnext(model_cfg)


def main():
    detector = model_init()
    img_dir = '/Users/lizhengyang/Desktop/record/'

    names = [n for n in os.listdir(img_dir) if n[-3:] in ['jpg', 'peg', 'bmp', 'png']]

    for n in names:
        detections = detector.perform_detect(img_dir + n, True, True)


if __name__ == '__main__':
    main()
