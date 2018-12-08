# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from keras.models import model_from_json
import json
import numpy as np


def deal_model_with_dw2conv(keras_json, keras_weight):

    if 'DepthwiseConv2D' not in open(keras_json).read():
        print("no DepthwiseConv2D")
        pass
    else:
        print("we need convert DepthwiseConv2D to  Conv2D")
        model = model_from_json(open(keras_json).read())
        model.load_weights(keras_weight, by_name=True)
        obj = json.loads(open(keras_json).read())
        layers = []
        dws = []
        for layer in obj['config']['layers']:
            if layer['class_name'] == 'DepthwiseConv2D':
                layer['class_name'] = 'Conv2D'
                filters = model.get_layer(layer['name']).get_weights()[0].shape[2]
                try:
                    depth_multiplier = layer['config']['depth_multiplier']
                except:
                    depth_multiplier = 1
                layer['config']['filters'] = filters * depth_multiplier
                try:
                    del layer['config']['depth_multiplier']
                except:
                    pass
                try:
                    del layer['config']['depthwise_initializer']
                except:
                    pass
                try:
                    del layer['config']['depthwise_regularizer']
                except:
                    pass
                try:
                    del layer['config']['depthwise_constraint']
                except:
                    pass
                dws.append(layer['name'])
            layers.append(layer)
        obj['config']['layers'] = layers
        model1 = model_from_json(json.dumps(obj))

        open(keras_json, 'w').write(json.dumps(obj, indent=4))

        for layer in model1.layers:
            if len(model.get_layer(layer.name).get_weights()) == 0:
                continue
            if layer.name in dws:
                w_b = model.get_layer(layer.name).get_weights()

                ws = []
                for depth in range(w_b[0].shape[3]):
                    w = w_b[0][:, :, :, depth]
                    for i in range(w.shape[2]):
                        one = np.zeros((w.shape[0], w.shape[1], w.shape[2]))
                        one[:, :, i] = w[:, :, i]
                        ws.append(one)
                wss = []
                for x in range(w_b[0].shape[2]):
                    for y in range(len(ws)):
                        if y % w_b[0].shape[2] == x:
                            wss.append(ws[y])

                nw = np.zeros((w_b[0].shape[0], w_b[0].shape[1], w_b[0].shape[2], w_b[0].shape[2] * w_b[0].shape[3]))
                for i, w in enumerate(wss):
                    nw[:, :, :, i] = wss[i]

                try:
                    b = w_b[1]
                    layer.set_weights([nw, b])
                except:
                    layer.set_weights([nw])
            else:
                layer.set_weights(model.get_layer(layer.name).get_weights())

        model1.save_weights(keras_weight)
