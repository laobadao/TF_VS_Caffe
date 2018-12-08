# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from quatization.lib.common import *
from keras.layers import *
from keras.models import Model
import keras.backend.tensorflow_backend as K
import numpy as np
import h5py
from quatization.lib.CustomLayer import *
import tensorflow as tf
from keras.layers.core import Layer


class QuantificationCore(Layer):
    def __init__(self, FI, FR, **kwargs):
        super(QuantificationCore, self).__init__(**kwargs)
        self.FI = FI
        self.FR = FR
        self.vq = 1 << FR
        self.vmin = -1 * (1 << (FI + FR))
        self.vmax = (1 << (FI + FR)) - 1

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        o1 = tf.to_int32(x * self.vq)
        o2 = tf.maximum(o1, self.vmin)
        o3 = tf.minimum(o2, self.vmax)
        o4 = tf.to_float(tf.div(tf.to_float(o3), float(self.vq)))
        return o4

    def get_config(self):
        config = {'FI': self.FI, 'FR': self.FR, 'trainable': 'false'}
        base_config = super(QuantificationCore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantificationCore8(Layer):
    def __init__(self, layer_max, mode, **kwargs):
        super(QuantificationCore8, self).__init__(**kwargs)
        self.layer_max = layer_max
        self.mode = mode
        if mode == "uint8":
            n = 8
        elif mode == "int8":
            n = 7
        else:
            raise RuntimeError('mode is uint8 or int8')
        if n == 7:
            self.vmin = -1 * (1 << n)
        else:
            self.vmin = 0
        self.vmax = (1 << n) - 1

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        o1 = (x / self.layer_max) * self.vmax
        o2 = tf.maximum(o1, self.vmin)
        o3 = tf.minimum(o2, self.vmax)
        if self.mode == 'uint8':
            o4 = tf.cast(o3, tf.uint8)
        if self.mode == 'int8':
            o4 = tf.cast(o3, tf.int8)
        o5 = tf.cast(o4, tf.float32)
        o6 = (o5 * self.layer_max) / self.vmax
        return o6

    def get_config(self):
        config = {'N': 8, 'trainable': 'false'}
        base_config = super(QuantificationCore8, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvPreprocess(DepthwiseConv2D):

    def add_weight(self, shape, **kwargs):
        print(shape)
        kernel = np.zeros(shape=[shape[0], shape[1], shape[2], 1])
        kernel[int((shape[0] - 1) / 2), int((shape[1] - 1) / 2), :, :] = 1
        return kernel.astype(np.float32)


class CustomDense(Dense):
    E_BLOCK_SIZE = 32

    def reconfig(self, mode, s, absent_segs):
        self.scaler = None
        self.mode = mode
        self.absent_segs = absent_segs
        if s is None:
            return
        t = np.copy(s)
        self.scaler = t

    def call(self, inputs):

        output = K.dot(inputs, self.kernel)

        if self.scaler is not None:
            output = tf.multiply(output, self.scaler)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output


class CustomConv2D(Conv2D):
    E_BLOCK_SIZE = 32

    def reconfig(self, mode, scale, edge, fast, conv_input_shape, conv_output_shape, absent_segs):

        self.dw_kernel = np.ones(shape=[1, 1, conv_input_shape[2], 1]).astype(np.float32)
        self.edge = 63
        self.fast = True;
        self.scale = None
        self.mode = mode
        if scale is None:
            return

        t = np.empty(shape=conv_output_shape)
        N = t.shape[2]
        for n in range(N):
            t[:, :, n] = scale[n]

        self.scale = t
        self.absent_segs = absent_segs

    def call(self, x, training=None):
        s = K.conv2d(x, kernel=self.kernel, strides=self.strides, padding='valid')
        # =========================================
        if self.use_bias:
            s = K.bias_add(
                s,
                self.bias,
                data_format=self.data_format)
        if self.scale is not None:
            s = tf.multiply(s, self.scale)
            # ===========================================
        return s


pool_class = {
    'max': MaxPooling2D,
    'avg': AveragePooling2D,
    'gavg': GlobalAveragePooling2D
}


def SuperLayerModel(super_layer, mode, model_parameter_file, edge_file, fast=False, layerto=None, absent_segs=False,
                    quant=None, bit='int8'):
    input_shape = super_layer['layers'][0]['input_shape']
    sinput = Input(shape=input_shape)
    next_seg = sinput
    layer_tensor = {
        super_layer['layers'][0]['input'][0][1]: sinput
    }

    if model_parameter_file is not None:
        model_parameter = H52DataSet(model_parameter_file)
    else:
        model_parameter = {};

    if edge_file is not None:
        edge_parameter = json.load(open(edge_file))
    else:
        edge_parameter = {}
    output = []
    if mode[:3] == 'MNF' and quant == None:
        raise RuntimeError('when mode is MNF or MNFIBF,please must input quant')
    if mode[:3] == 'MNF' and quant.endswith(".json"):
        quants_layer = json.loads(open(quant).read())
    if mode[:3] == 'MNF' and quant.endswith(".h5"):
        quants_layer = h5py.File(quant)

    layers_map, super_order = {}, []
    def set_sublayer_info(super_layer, sub_layer):
        layers_map.setdefault(sub_layer, {'super_type': super_layer['type'],
            'super_name': super_layer['name']})

    for i, layer in enumerate(super_layer['layers']):

        if layerto is not None and i > layerto:
            break
        output_ref_name = layer['output_ref_name']
        input_shape = layer['input_shape']
        last_layer = None

        super_order.append(layer['name'])
        if layer['type'] in ['Conv2D', 'AveragePooling2D', 'MaxPooling2D', 'Copy', 'Dense', 'DepthwiseConv2D']:

            next_seg = layer_tensor[layer['input'][0][1]]

            if layer['type'] in ['Conv2D', 'DepthwiseConv2D']:
                if mode == 'MZ':
                    use_bias = (layer['conv_bias'] == 1)
                else:
                    use_bias = True

                next_seg = ZeroPadding2D(padding=layer['conv_padding'])(next_seg)

                if mode in ['MZ']:
                    ConvClass = Conv2D
                else:
                    ConvClass = CustomConv2D
                last_layer = ConvClass(
                    layer['conv_kernel'][2],
                    layer['conv_kernel'][:2],
                    activation='linear',
                    padding='valid',
                    strides=layer['conv_strides'],
                    use_bias=use_bias,
                    name=layer['conv_layer_name'])

                if mode != 'MZ':
                    last_layer.reconfig(
                        mode,
                        model_parameter.get(layer['name']),
                        edge_parameter.get(layer['output_ref_name'] + '/data', [0, 63])[1],
                        fast,
                        layer['input_shape'],
                        layer['conv_output_shape'],
                        absent_segs
                    )
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            if layer['type'] == 'Dense':
                next_seg = Reshape(
                    [input_shape[-1]],
                    name='rs%d' % i
                )(next_seg)

                if mode in ['MZ']:
                    DenseClass = Dense
                else:
                    DenseClass = CustomDense
                last_layer = DenseClass(
                    layer['conv_kernel'][2],
                    activation='linear',
                    name=layer['dense_layer_name'])
                if mode != 'MZ':
                    last_layer.reconfig(mode, model_parameter.get(layer['name']), absent_segs)
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            if layer['bn'] and mode == 'MZ':
                last_layer = BatchNormalization(axis=3, scale=layer.get('bn_scale', False), name=layer['bn_layer_name'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            if layer['activation'] in ['relu', 'softmax', 'linear', 'sigmoid', 'leakyrelu']:
                last_layer = Activation(layer['activation'], name='act_' + layer['output_ref_name'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            if layer['activation'] in ['PReLU', 'prelu']:
                last_layer = PReLU(shared_axes=[1, 2], name=layer['output_ref_name'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            if layer['pool_type'] in pool_class.keys():
                PoolClass = pool_class[layer['pool_type']]
                if layer['pool_type'] == 'gavg':
                    last_layer = PoolClass(name=layer['pool_layer_name'])
                    next_seg = last_layer(next_seg)
                    set_sublayer_info(layer, last_layer)
                else:
                    padd = layer.get('pool_padding_repr', 'valid')
                    if padd == 'custom':
                        next_seg = ZeroPadding2D(padding=layer['pool_padding'])(next_seg)
                        padd = 'valid'
                    last_layer = PoolClass(
                        layer['pool_size'],
                        strides=layer['pool_strides'],
                        padding=padd,
                        name=layer['pool_layer_name'])
                    next_seg = last_layer(next_seg)
                    set_sublayer_info(layer, last_layer)

            if 'reshape' in layer.keys():
                last_layer = Reshape(layer['reshape'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            layer_tensor[layer['name']] = next_seg
            last_layer.name = 'out_' + layer['output_ref_name']

        elif layer['type'] == 'Concatenate':
            '''
            if layer['axis'] not in [-1,3]:
                print('[ERROR] Concatenate with none-channel is not supported')
                return None
            '''
            input_tensors = []
            for i, input_config in enumerate(layer['input']):
                input_tensors.append(layer_tensor[input_config[1]])
            last_layer = Concatenate(-1, name=layer['output_ref_name'])
            next_seg = last_layer(input_tensors)
            set_sublayer_info(layer, last_layer)

            try:
                last_layer = Activation(layer['activation'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)
            except:
                pass

            layer_tensor[layer['name']] = next_seg
            last_layer.name = 'out_' + layer['output_ref_name']

        elif layer['type'] == 'Add':
            input_tensors = []
            for i, input_config in enumerate(layer['input']):
                input_tensors.append(layer_tensor[input_config[1]])

            last_layer = Add(name=layer['name'])
            next_seg = last_layer(input_tensors)
            set_sublayer_info(layer, last_layer)

            if layer['activation'] in ['relu', 'softmax', 'linear', 'leakyrelu']:
                last_layer = Activation(layer['activation'], name='act_' + layer['output_ref_name'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            layer_tensor[layer['name']] = next_seg
            last_layer.name = 'out_' + layer['output_ref_name']

        elif layer['type'] == 'SpaceToDepth':
            last_layer = SpaceToDepth(block_size=layer['block_size'])
            next_seg = last_layer(layer_tensor[layer['input'][0][1]])
            set_sublayer_info(layer, last_layer)

            layer_tensor[layer['name']] = next_seg
            last_layer.name = 'out_' + layer['output_ref_name']

        elif layer['type'] == 'UpSampling2D':
            last_layer = UpSampling2D(size=tuple(layer['size']))
            next_seg = last_layer(layer_tensor[layer['input'][0][1]])
            set_sublayer_info(layer, last_layer)

            layer_tensor[layer['name']] = next_seg
            last_layer.name = 'out_' + layer['output_ref_name']

        elif layer['type'] == 'Permute':
            last_layer = Permute(dims=layer['dims'])
            next_seg = last_layer(layer_tensor[layer['input'][0][1]])
            if "activation" in layer.keys():
                last_layer = Activation(layer['activation'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            if "reshape" in layer.keys():
                last_layer = Reshape(target_shape=layer['reshape'])
                next_seg = last_layer(next_seg)
                set_sublayer_info(layer, last_layer)

            layer_tensor[layer['name']] = next_seg
            last_layer.name = 'out_' + layer['output_ref_name']
        else:
            print('[ERROR] Not supported layer type', layer['type'])

        if mode[:3] == 'MNF' and quant != None and quant.endswith(".json") and bit == 'bit16':
            q = int(quants_layer[layer['name']].split('_')[1])
            last_layer = QuantificationCore(q, 15 - q)
            next_seg = last_layer(next_seg)
            set_sublayer_info(layer, last_layer)

            layer_tensor[layer['name']] = next_seg
            print('layer', layer['name'], '1_{}_{}'.format(q, 15 - q))

        elif mode[:3] == 'MNF' and quant != None and quant.endswith(".h5") and bit in ['int8', 'uint8']:
            last_layer = QuantificationCore8(np.array(quants_layer[layer['name']]), bit)
            next_seg = last_layer(next_seg)
            set_sublayer_info(layer, last_layer)

            layer_tensor[layer['name']] = next_seg
            print('layer', layer['name'], bit)

        if layer['is_output'] == 1:
            output.append(next_seg)

    lname_map = {'_super_order' : super_order}
    for layer, info in layers_map.items():
        lname_map[layer.name] = info

    model = Model(sinput, output, name='ISD')
    return model, lname_map
