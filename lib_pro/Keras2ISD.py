#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from quatization.lib.common import *
import json
import sys
import os
import math
import functools
import operator
from quatization.lib.CustomLayer import *
from keras.models import model_from_json


def GetPaddingSizeAndOutputShape(padding_repr, in_height, in_width, strides, filter_height, filter_width):
    strides = [1] + strides

    if "[" in padding_repr:
        padding_repr = padding_repr.split("/")[1].replace(" ", "")[1:-1].split(",")

        top = int(padding_repr[0])
        bottom = int(padding_repr[1])
        left = int(padding_repr[2])
        right = int(padding_repr[3])

        out_height = int((in_height + top + bottom - filter_height) / strides[1]) + 1
        out_width = int((in_width + left + right - filter_width) / strides[2]) + 1
        padding = ((top, bottom), (left, right))

        return [out_height, out_width], [[top, bottom], [left, right]]

    if padding_repr == 'same':

        out_height = math.ceil(float(in_height) / float(strides[1]))
        out_width = math.ceil(float(in_width) / float(strides[2]))

        if in_height % strides[1] == 0:
            pad_along_height = max(filter_height - strides[1], 0)
        else:
            pad_along_height = max(filter_height - (in_height % strides[1]), 0)

        if in_width % strides[2] == 0:
            pad_along_width = max(filter_width - strides[2], 0)
        else:
            pad_along_width = max(filter_width - (in_width % strides[2]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        return [int(out_height), int(out_width)], [[pad_top, pad_bottom], [pad_left, pad_right]]

    else:
        out_height = math.ceil(float(in_height - filter_height + 1) / float(strides[1]))
        out_width = math.ceil(float(in_width - filter_width + 1) / float(strides[2]))

        return [int(out_height), int(out_width)], [[0, 0], [0, 0]]


def insertLayer(json_path):
    """
    if input_layer is other multiple layer's input ,need to add a linear activation

    :param json_path:
    :return:
    """
    js = open(json_path).read()
    json_obj = json.loads(js)
    # input layer name
    input_name = json_obj['config']['layers'][0]['name']
    num = 0
    for obj in json_obj['config']['layers'][1:]:
        if obj['inbound_nodes'][0][0][0] == input_name:
            num += 1

    if num > 1:
        print("=" * 100)
        json_obj = json.loads(js)
        layers = json_obj['config']['layers']
        s = '{"inbound_nodes": [[["' + input_name + '", 0, 0, {}]]],"name": "frcnn_second","config": {"name": "frcnn_second","activation": "linear","trainable": true},"class_name": "Activation"}'
        layers.insert(1, json.loads(s))

        for layer in layers[2:]:
            inputs = layer['inbound_nodes'][0][0][0]
            if inputs == input_name:
                layer['inbound_nodes'][0][0][0] = "frcnn_second"

        json_obj['config']['layers'] = layers
        open(json_path, "w").write(json.dumps(json_obj, indent=4))
    else:
        print("---------------------------do nothing -----------------------------")


def Keras2ISD(input_file, mode, input_size=None):
    insertLayer(input_file)
    json_obj = json.load(open(input_file))
    super_layers = {
        'config': {
            'input_mode': mode
        },
        'layers': []
    }
    try:
        layers_obj = json_obj['config']['layers']
    except:
        layers_obj = json_obj['config']

    # =====================del dropout============================
    dropout = []

    for i, obj in enumerate(layers_obj):

        if obj['class_name'] == 'Dropout':
            dropout = [obj['name'], obj['inbound_nodes'][0][0][0]]
            del layers_obj[i]
            break

    # 找到输入为 dropout 的 layer ,替换为原 dropout 的前一层，删除 dropout 后，将模型前后连接起来
    if len(dropout) != 0:
        for obj in layers_obj[1:]:
            if obj['inbound_nodes'][0][0][0] == dropout[0]:
                obj['inbound_nodes'][0][0][0] = dropout[1]

    # ==================== del ZeroPadding2D before pooling=============================
    need_del = []

    for i, layer in enumerate(layers_obj):
        # caffe model has ZeroPadding before pooling, delete ZeroPadding and add another attr to pooling
        if layer['class_name'] == 'ZeroPadding2D' \
                and layers_obj[i + 1]['class_name'] in ['MaxPooling2D', 'AveragePooling2D']:
            input_name = layer['inbound_nodes'][0][0][0]
            name = layer['config']['name']
            padding = layer['config']['padding']
            need_del.append(layer)
            layers_obj[i + 1]['inbound_nodes'][0][0][0] = input_name
            # 在 pooling 层中自定义添加的 padding 属性
            layers_obj[i + 1]['padding'] = padding

    for obj in need_del:
        layers_obj.remove(obj)

    # ====================get all layers  output_shape=============================
    all_layers_output = {}
    model = model_from_json(open(input_file).read(), custom_objects={'SpaceToDepth': SpaceToDepth})
    for layer in model.layers:
        # eg: output_shape: (12, 12, 384)
        output_shape = layer.output_shape[1:]
        if len(output_shape) == 1:
            output_shape = [1, 1, output_shape[0]]

        all_layers_output[layer.name] = [int(i) for i in output_shape]
    # =================================================

    try:
        super_layers['config']['name'] = json_obj['config']['name']
    except:
        super_layers['config']['name'] = 'Model'

    input_size_array = None

    if input_size is None:
        for layer in layers_obj:
            layer_class_name = layer['class_name']
            if layer_class_name == 'InputLayer':
                input_size_array = layer['config']['batch_input_shape'][1:]
                break

    else:
        input_size_array = input_size.split('x')
        if len(input_size_array) == 1:
            input_size_array = [int(input_size_array[0]), int(input_size_array[0]), 3]
        elif len(input_size_array) == 2:
            input_size_array = [int(input_size_array[0]), int(input_size_array[1]), 3]
        else:
            input_size_array = input_size_array[0:3]

    if input_size_array is None:
        return (False, "Invalid input shape")

    layer_name_fast_cache = {}
    name_config_dict = {}
    # find all inputs for all layers
    inputss = []

    for layer1 in layers_obj[1:]:
        for ins in layer1['inbound_nodes'][0]:
            inputss.append(ins[0])

    for i, layer in enumerate(layers_obj):
        # layer 的名字在两个地方都有，是一样的
        try:
            layer_name = layer['name']
        except:
            layer_name = layer['config']['name']

        layer_class_name = layer['class_name']
        layer_config = layer['config']
        layer_input = layer['inbound_nodes']

        name_config_dict[layer_name] = [layer_class_name, layer_config, layer_input]
        if i == 1 and layers_obj[1]['class_name'] == 'Activation' and layers_obj[1]['name'] == 'frcnn_second' and \
                layers_obj[1]['config']['activation'] == 'linear':
            super_layer_obj = {
                'type': 'Copy',
                'name': layer_name,
                'output_ref_name': layer_name,
                'pool_layer_name': 'none',
                'conv_bias': 0,
                'conv_padding_repr': "valid",
                'conv_padding': [[0, 0], [0, 0]],
                'conv_kernel': [1, 1, 1],
                'conv_strides': [1, 1],
                'raw_input': layer_input,
                'is_output': 0,
                'bn': 0,
                'activation': 'linear',
                'pool_type': 'none',
                'pool_size': [0, 0],
                'pool_strides': [0, 0],
                'pool_padding_repr': 'none',
                'pool_padding': [[0, 0], [0, 0]]

            }

            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1
            continue

        if i == 1 and layers_obj[1]['class_name'] == 'Flatten':
            super_layer_obj = {
                'type': 'Copy',
                'name': layer_name,
                'output_ref_name': layer_name,
                'pool_layer_name': 'none',
                'conv_bias': 0,
                'conv_padding_repr': "valid",
                'conv_padding': [[0, 0], [0, 0]],
                'conv_kernel': [1, 1, 1],
                'conv_strides': [1, 1],
                'raw_input': layer_input,
                'is_output': 0,
                'bn': 0,
                'activation': 'linear',
                'pool_type': 'none',
                'pool_size': [0, 0],
                'pool_strides': [0, 0],
                'pool_padding_repr': 'none',
                'pool_padding': [[0, 0], [0, 0]],
                'reshape': [-1]

            }
            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1
            continue

        if layer_class_name in ['InputLayer', 'ZeroPadding2D']:
            pass

        elif layer_class_name == 'Conv2D' or layer_class_name == 'DepthwiseConv2D':

            kernel_size = layer_config['kernel_size']
            if layer_class_name == 'Conv2D':
                # filters 是 C channel 方向的格式，也就是 filter 的个数
                kernel_size.append(layer_config['filters'])

            super_layer_obj = {
                'type': layer_class_name,
                'name': layer_name,
                'output_ref_name': layer_name,
                'conv_layer_name': layer_name,
                'conv_bias': 1 if layer_config['use_bias'] else 0,
                'conv_padding_repr': layer_config['padding'],
                'conv_padding': None,
                'conv_kernel': kernel_size,
                'conv_strides': layer_config['strides'],
                'raw_input': layer_input,
                'is_output': 0,
                'bn': 0,
                'activation': layer_config.get('activation', 'linear'),
                'pool_type': 'none'
            }

            input_name = layer_input[0][0][0]

            if name_config_dict[input_name][0] == 'ZeroPadding2D':
                # conv_padding_repr 参数被废弃，NPU 不会取该参数， 取 same 或者 valid，由于算法歧义而废止
                super_layer_obj['conv_padding_repr'] = 'custom'
                super_layer_obj['conv_padding'] = name_config_dict[input_name][1]['padding']
                super_layer_obj['raw_input'] = name_config_dict[input_name][2]

            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name == 'Dense':
            kernel_size = [1, 1, layer_config['units']]
            padding = [[0, 0], [0, 0]]
            super_layer_obj = {
                'type': layer_class_name,
                'name': layer_name,
                'output_ref_name': layer_name,
                'dense_layer_name': layer_name,
                'conv_bias': 1 if layer_config['use_bias'] else 0,
                'conv_padding_repr': 'valid',
                'conv_padding': padding,
                'conv_kernel': kernel_size,
                'conv_strides': [1, 1],
                'raw_input': layer_input,
                'is_output': 0,
                'bn': 0,
                'activation': layer_config.get('activation', 'linear'),
                'pool_type': 'none'

            }

            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name == 'BatchNormalization':

            input_layer = layer['inbound_nodes'][0][0][0]
            cached_id = layer_name_fast_cache[input_layer]
            # 判断该层 layer 是否属于分支中的一条
            brothes = inputss.count(input_layer)

            if (super_layers['layers'][cached_id]['type'] in ['Conv2D', 'DepthwiseConv2D']
                    and super_layers['layers'][cached_id]['bn'] == 0 and brothes < 2):

                # bn bn_scale bn_center bn_layer_name 该参数已被废弃
                super_layers['layers'][cached_id]['bn'] = 1
                super_layers['layers'][cached_id]['bn_scale'] = layer_config.get('scale', False)
                super_layers['layers'][cached_id]['bn_center'] = layer_config.get('center', False)
                super_layers['layers'][cached_id]['output_ref_name'] = layer_name
                super_layers['layers'][cached_id]['bn_layer_name'] = layer_name
                layer_name_fast_cache[layer_name] = cached_id
            else:
                # 如果 brothes >= 2 则代表 上一层往下有多个分支 所有分支都需要另起一层 作为单独的 ISD
                super_layer_obj = {
                    'type': 'Copy',
                    'name': layer_name,
                    'output_ref_name': layer_name,
                    'pool_layer_name': 'none',
                    'conv_bias': 0,
                    'conv_padding_repr': "valid",
                    'conv_padding': [[0, 0], [0, 0]],
                    'conv_kernel': [1, 1, 1],
                    'conv_strides': [1, 1],
                    'raw_input': layer_input,
                    'is_output': 0,
                    'bn': 1,
                    'activation': layer_config.get('activation', 'linear'),
                    'pool_type': 'none',
                    'bn_scale': False,
                    'bn_layer_name': layer_name,
                    'bn_center': True

                }

                super_layers['layers'].append(super_layer_obj)
                layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name in ['MaxPooling2D', 'AveragePooling2D']:

            input_layer = layer['inbound_nodes'][0][0][0]
            cached_id = layer_name_fast_cache[input_layer]

            if layer_class_name == 'MaxPooling2D':
                pool_type = 'max'
            else:
                pool_type = 'avg'
            brothes = inputss.count(input_layer)  # count brother node
            if 'padding' in layer.keys():
                pool_padding_repr = 'custom'
                pool_padding = layer['padding']
            else:
                pool_padding_repr = layer_config['padding']
                pool_padding = None

            if (super_layers['layers'][cached_id]['type'] in ['Conv2D', 'DepthwiseConv2D', 'Copy']
                    and super_layers['layers'][cached_id]['pool_type'] == 'none' and brothes < 2):

                super_layers['layers'][cached_id]['pool_type'] = pool_type
                super_layers['layers'][cached_id]['pool_padding'] = pool_padding
                super_layers['layers'][cached_id]['pool_padding_repr'] = pool_padding_repr
                super_layers['layers'][cached_id]['pool_size'] = layer_config['pool_size']
                super_layers['layers'][cached_id]['pool_strides'] = layer_config['strides']
                super_layers['layers'][cached_id]['output_ref_name'] = layer_name
                super_layers['layers'][cached_id]['pool_layer_name'] = layer_name
                layer_name_fast_cache[layer_name] = cached_id

            else:
                super_layer_obj = {
                    'type': 'Copy',
                    'name': layer_name,
                    'output_ref_name': layer_name,
                    'pool_layer_name': layer_name,
                    'conv_bias': 0,
                    'conv_padding_repr': "valid",
                    'conv_padding': [[0, 0], [0, 0]],
                    'conv_kernel': [1, 1, 1],
                    'conv_strides': [1, 1],
                    'raw_input': layer_input,
                    'is_output': 0,
                    'bn': 0,
                    'activation': layer_config.get('activation', 'linear'),
                    'pool_type': pool_type,
                    'pool_size': layer_config['pool_size'],
                    'pool_strides': layer_config['strides'],
                    'pool_padding_repr': pool_padding_repr,
                    'pool_padding': pool_padding
                }
                super_layers['layers'].append(super_layer_obj)
                # 每一个超级层都有唯一的 id
                layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name == 'Activation':
            input_layer = layer['inbound_nodes'][0][0][0]
            cached_id = layer_name_fast_cache[input_layer]
            # 给超级层以卷积分核心的层添加激活函数
            super_layers['layers'][cached_id]['activation'] = layer_config['activation']
            super_layers['layers'][cached_id]['output_ref_name'] = layer_name
            layer_name_fast_cache[layer_name] = cached_id

        elif layer_class_name in ['LeakyReLU', 'PReLU']:
            input_layer = layer['inbound_nodes'][0][0][0]
            cached_id = layer_name_fast_cache[input_layer]
            super_layers['layers'][cached_id]['activation'] = layer_class_name.lower()
            super_layers['layers'][cached_id]['output_ref_name'] = layer_name
            layer_name_fast_cache[layer_name] = cached_id

        elif layer_class_name == 'GlobalAveragePooling2D':

            super_layer_obj = {

                'type': 'Copy',
                'name': layer_name,
                'output_ref_name': layer_name,
                'pool_layer_name': layer_name,
                'conv_bias': 0,
                'conv_padding_repr': 'valid',
                'conv_padding': [[0, 0], [0, 0]],
                'conv_kernel': [1, 1, 1],
                'conv_strides': [1, 1],
                'raw_input': layer_input,
                'is_output': 0,
                'bn': 0,
                'activation': layer_config.get('activation', 'linear'),
                'pool_type': 'gavg',
                'pool_size': [0, 0],
                'pool_strides': [0, 0]

            }

            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name in ['Concatenate', 'PriorBox', 'Add', 'Dropout', 'Merge']:

            if layer_class_name == 'Merge' and layer_config['mode'] == 'concat':
                layer_class_name = 'Concatenate'
                layer_config['axis'] = layer_config['concat_axis']

            super_layer_obj = {
                'name': layer_name,
                'type': layer_class_name,
                'output_ref_name': layer_name,
                'raw_input': layer_input,
                'is_output': 0,
                'pool_type': 'none',
                'activation': 'linear'
            }

            if layer_class_name == 'Concatenate':
                print(all_layers_output[layer_name])

                if layer_config['axis'] < 0:
                    layer_config['axis'] += len(all_layers_output[layer_name]) + 1

                super_layer_obj['axis'] = layer_config['axis'] - 1
            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name in ['Reshape', 'Flatten']:

            input_layer = layer['inbound_nodes'][0][0][0]
            cached_id = layer_name_fast_cache[input_layer]
            if layer_class_name == 'Reshape':
                super_layers['layers'][cached_id]['reshape'] = layer_config['target_shape']
            if layer_class_name == 'Flatten':
                super_layers['layers'][cached_id]['reshape'] = [-1, ]
            super_layers['layers'][cached_id]['output_ref_name'] = layer_name
            layer_name_fast_cache[layer_name] = cached_id

        elif layer_class_name == 'UpSampling2D':

            super_layer_obj = {
                'name': layer_name,
                'type': layer_class_name,
                'output_ref_name': layer_name,
                'raw_input': layer_input,
                'is_output': 0,
                'pool_type': 'none',
                'size': layer_config['size'],
                'activation': 'linear'

            }

            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name == 'SpaceToDepth':

            super_layer_obj = {
                'name': layer_name,
                'type': layer_class_name,
                'output_ref_name': layer_name,
                'raw_input': layer_input,
                'is_output': 0,
                'pool_type': 'none',
                'block_size': layer_config['block_size'],
                'activation': 'linear'
            }

            super_layers['layers'].append(super_layer_obj)

            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1

        elif layer_class_name == 'Permute':

            super_layer_obj = {
                'name': layer_name,
                'type': layer_class_name,
                'output_ref_name': layer_name,
                'raw_input': layer_input,
                'is_output': 0,
                'activation': 'linear',
                'dims': layer['config']['dims']
            }

            super_layers['layers'].append(super_layer_obj)
            layer_name_fast_cache[layer_name] = len(super_layers['layers']) - 1
        else:
            print("[ERROR] Not supported layer", layer_class_name)

    cache_output_ref_name_to_layer = {}

    for i, layer in enumerate(super_layers['layers']):
        cache_output_ref_name_to_layer[layer['output_ref_name']] = [i, layer['name']]
        if layer['type'] in ['Concatenate', 'Add']:
            cache_output_ref_name_to_layer[layer['name']] = [i, layer['name']]

    for i, layer in enumerate(super_layers['layers']):
        inputs = []
        if layer['type'] in ['Concatenate', 'Add', 'Dropout']:
            inputs = [r[0] for r in layer['raw_input'][0]]
        else:
            inputs = [layer['raw_input'][0][0][0]]
        inputs = [cache_output_ref_name_to_layer.get(r, [-1, r]) for r in inputs]

        if layer['type'] == 'PriorBox':
            inputs = [[-1, '']]

        layer['input'] = inputs
    has_output_cache = {}

    for i, layer in enumerate(super_layers['layers']):
        for item in layer['input']:
            has_output_cache[item[1]] = 1

    for i, layer in enumerate(super_layers['layers']):
        if layer['name'] not in has_output_cache.keys():
            layer['is_output'] = 1

    if isinstance(model.output, list):
        has_output_cache = [layer.name.split('/')[0] for layer in model.output]
    else:
        has_output_cache = []

    for output in has_output_cache:
        for i, layer in enumerate(super_layers['layers']):

            if layer['output_ref_name'] in has_output_cache and layer['is_output'] == 0:
                layer['is_output'] = 1

    for i, layer in enumerate(super_layers['layers']):
        layer['id'] = i
        layer_type = layer['type']
        if i == 0:
            layer['input_shape'] = input_size_array

        elif layer_type in ['Conv2D', 'DepthwiseConv2D', 'Dense', 'Copy', 'Concatenate', 'Add', 'AveragePooling2D',
                            'Dropout', 'MaxPool2D', 'UpSampling2D', 'SpaceToDepth', 'Permute']:

            layer['input_shape'] = all_layers_output[layer['raw_input'][0][0][0]]

            if layer_type == 'Dense':
                layer['input_shape'] = [1, 1, functools.reduce(operator.__mul__, layer['input_shape'])]

            if layer_type == 'Concatenate':
                # print(layer['input_shape'],layer['axis'])
                layer['input_shape'][layer['axis']] = 0

        if layer_type in ['Conv2D', 'DepthwiseConv2D', 'Dense', 'Copy', 'AveragePooling2D', 'MaxPool2D']:
            if len(layer['input_shape']) == 1:
                output_shape, padding = all_layers_output[layer['output_ref_name']], [[0, 0], [0, 0]]
            else:
                output_shape, padding = GetPaddingSizeAndOutputShape(
                    layer['conv_padding_repr'],
                    layer['input_shape'][0],
                    layer['input_shape'][1],
                    layer['conv_strides'],
                    layer['conv_kernel'][0],
                    layer['conv_kernel'][1])
            layer['conv_padding_repr'] = layer['conv_padding_repr'].split("/")[0]

            if not layer['conv_padding']:
                layer['conv_padding'] = list(padding)

            layer['output_shape'] = [int(i) for i in all_layers_output[layer['output_ref_name']]]

            if layer_type in ['Conv2D', 'DepthwiseConv2D']:
                layer['conv_output_shape'] = all_layers_output[layer['conv_layer_name']]
            if layer_type in ['Dense']:
                layer['conv_output_shape'] = all_layers_output[layer['dense_layer_name']]

            if layer['pool_type'] in ['max', 'avg']:
                pool_input_shape = [int(i) for i in model.get_layer(layer['pool_layer_name']).input_shape[1:]]
                output_shape, padding = GetPaddingSizeAndOutputShape(
                    layer['pool_padding_repr'],
                    pool_input_shape[0],
                    pool_input_shape[1],
                    layer['pool_strides'],
                    layer['pool_size'][0],
                    layer['pool_size'][1]
                )

                layer['pool_padding_repr'] = layer['pool_padding_repr'].split("/")[0]

                if layer['pool_padding'] == None:
                    layer['pool_padding'] = list(padding)

        elif layer_type == 'Concatenate':

            layer['output_shape'] = [int(i) for i in all_layers_output[layer['output_ref_name']]]
            if layer['pool_type'] in ['max', 'avg']:
                pool_input_shape = [int(i) for i in model.get_layer(layer['pool_layer_name']).input_shape[1:]]
                output_shape, padding = GetPaddingSizeAndOutputShape(
                    layer['pool_padding_repr'],
                    pool_input_shape[0],
                    pool_input_shape[1],
                    layer['pool_strides'],
                    layer['pool_size'][0],
                    layer['pool_size'][1]
                )
                layer['pool_padding'] = padding
        elif layer_type in ['Add', 'Dropout']:
            layer['output_shape'] = [int(i) for i in all_layers_output[layer['output_ref_name']]]
        elif layer_type == 'UpSampling2D':
            layer['output_shape'] = [int(i) for i in all_layers_output[layer['output_ref_name']]]
        elif layer_type == 'SpaceToDepth':
            layer['output_shape'] = [int(i) for i in all_layers_output[layer['output_ref_name']]]

        elif layer_type == 'Permute':
            layer['output_shape'] = [int(i) for i in all_layers_output[layer['output_ref_name']]]

    for i, layer in enumerate(super_layers['layers']):
        if layer['type'] != 'Concatenate' and (layer['input_shape'][0] == 0 or layer['output_shape'][0] == 0):
            print(layer['input_shape'], layer['output_shape'], layer['name'], "ERROR")
            return (False, "[ERROR] Check", i, layer['name'])
    return (True, super_layers)


def keras2isd(input_file, output_file, mode):

    if mode is None:
        mode = 'tf'
    t, super_layers = Keras2ISD(input_file, mode)
    if t:
        fp = open(output_file, "w")
        fp.write(json.dumps(super_layers, indent=4))
        fp.close()
    else:
        print(super_layers)
    d = json.loads(open(output_file).read())

    # os.system("rm " + output_file)

    for layer in d["layers"]:
        if layer["type"] == "DepthwiseConv2D":
            layer["type"] = "Conv2D"
            layer["dw"] = 1
            layer["conv_kernel"].append(layer["output_shape"][2])

    with open(output_file, "w") as f:
        f.write(json.dumps(d, indent=4))


if __name__ == '__main__':

    rtn = GetOpt(
        sys.argv[0],
        sys.argv[1:],
        'input_file=s',
        'output_file=s',
        'size_input:s',
        'mode:s'
    )

    if rtn is not None:
        input_file = rtn['input_file']
        output_file = rtn['output_file']
        input_size = rtn['size_input']
        mode = rtn['mode']

        if mode is None:
            mode = 'tf'
        t, super_layers = Keras2ISD(input_file, mode, input_size)
        if t:
            fp = open(output_file, "w")
            fp.write(json.dumps(super_layers, indent=4))
            fp.close()
        else:
            print(super_layers)
        d = json.loads(open(output_file).read())
        os.system("rm " + output_file)
        for layer in d["layers"]:
            if layer["type"] == "DepthwiseConv2D":
                layer["type"] = "Conv2D"
                layer["dw"] = 1
                layer["conv_kernel"].append(layer["output_shape"][2])

        with open(output_file, "w") as f:
            f.write(json.dumps(d, indent=4))
