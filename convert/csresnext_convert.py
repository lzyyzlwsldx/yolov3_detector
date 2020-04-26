import os
import io
import configparser
from collections import defaultdict
from absl import (app, flags)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Conv2D, Input, ZeroPadding2D, Add, UpSampling2D, MaxPooling2D,
                                     Concatenate, LeakyReLU, BatchNormalization)

flags.DEFINE_string('config_path', None, 'Path to darknet cfg file.')
flags.DEFINE_string('weights_path', None, 'Path to Darknet weights file.')
flags.DEFINE_string('output_path', './keras.h5', 'Path to output keras model file.')

FLAGS = flags.FLAGS


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def compose_conv(filters,
                 size,
                 stride,
                 padding,
                 batch_normalize,
                 weight_decay,
                 conv_weights,
                 prev_layer):
    conv_layer = (Conv2D(
        filters, (size, size),
        strides=(stride, stride),
        kernel_regularizer=l2(weight_decay),
        use_bias=not batch_normalize,
        weights=conv_weights,
        activation=None,
        padding=padding))(prev_layer)
    return conv_layer


def _main(_args):
    config_path = os.path.expanduser(FLAGS.config_path)
    weights_path = os.path.expanduser(FLAGS.weights_path)
    output_path = os.path.expanduser(FLAGS.output_path)
    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)
    assert output_path.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_path)
    # output_root = os.path.splitext(output_path)[0]

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(shape=(3,), dtype='int32', buffer=weights_file.read(12))
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))

    print('Weights Header: ', major, minor, revision, seen)
    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model.')
    input_layer = Input(shape=(None, None, 3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    count = 0
    out_index = []
    for section in cfg_parser.sections():
        print('*' * 50)
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            # pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            # padding = 'same' if pad == 1 and stride == 1 else 'valid'
            padding = 'same'
            groups = int(cfg_parser[section]['groups']) if 'groups' in cfg_parser[section].keys() else 1

            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))
            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = prev_layer.get_shape()
            weights_shape = (size, size, prev_layer_shape[-1] // groups, filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)
            print(prev_layer_shape)
            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)
            print('weights_size', weights_size, 'groups', groups, 'filters', filters)
            conv_bias = np.ndarray(shape=(filters,), dtype='float32', buffer=weights_file.read(filters * 4))
            count += filters

            bn_weight_list = []
            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]
            else:
                conv_bias = np.split(conv_bias, groups, axis=0)

            conv_weights = np.ndarray(shape=darknet_w_shape,
                                      dtype='float32',
                                      buffer=weights_file.read(weights_size * 4))

            count += weights_size
            if stride > 1:
                padding = 'valid'
                if size == 3:
                    prev_layer = ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
                elif size == 7:
                    prev_layer = ZeroPadding2D(((3, 2), (3, 2)))(prev_layer)

            # Darknet uses left and top padding instead of 'same' mode
            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            prev_layers = tf.split(prev_layer, groups, axis=3)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            print(conv_weights.shape)
            conv_weights = np.split(conv_weights, groups, axis=3)

            conv_outs = []
            for i in range(groups):
                cw = [conv_weights[i]] if batch_normalize else [conv_weights[i], conv_bias[i]]
                conv_outs.append(compose_conv(filters // groups, size, stride, padding, batch_normalize, weight_decay,
                                              cw, prev_layers[i]))
            if len(conv_outs) > 1:
                conv_layer = tf.concat(conv_outs, axis=3)
            else:
                conv_layer = conv_outs[0]
            if batch_normalize:
                conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                print(skip_layer)
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            if all_layers[index].get_shape()[-1] != prev_layer.get_shape()[-1]:
                calculation_part, side_part = tf.split(prev_layer, 2, axis=-1)
                short_cut_layer = Add()([calculation_part, all_layers[index]])
                short_cut_layer = tf.concat([short_cut_layer, side_part], axis=-1)
            else:
                short_cut_layer = Add()([all_layers[index], prev_layer])
            if activation == 'leaky':
                short_cut_layer = LeakyReLU(alpha=0.1)(short_cut_layer)
            all_layers.append(short_cut_layer)
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass
        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    not len(out_index) and out_index.append(len(all_layers) - 1)
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])
    print(model.summary())
    model.save('{}'.format(output_path))
    print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))
    remaining_weights > 0 and print('Warning: {} unused weights'.format(remaining_weights))


if __name__ == '__main__':
    flags.mark_flag_as_required('config_path')
    flags.mark_flag_as_required('weights_path')
    app.run(_main)
