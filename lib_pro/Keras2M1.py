#!/usr/bin/env python3

from __future__ import division
from __future__ import absolute_import
from quatization.lib.common import *
import numpy as np
import json
import math


def keras2M1(input_isd_file, input_keras_weight_file, output_file, mode, ref_output_file=None, quant=None):

    if ref_output_file is None:
        ref_output_file = output_file.replace(".h5", ".ref.h5")

    if mode is None:
        mode = 'MZ'

    kreader = KerasWeightReader(input_isd_file, input_keras_weight_file)
    kwriter = KerasWeightWriter()
    isd_model = json.load(open(input_isd_file))

    ref_data = {}
    total_size = 0
    is_first_conv = True

    if mode[:3] == 'MNF':
        if quant:
            weights_quant = json.loads(open(quant).read())
        else:
            raise RuntimeError("please input quant file")
    i = 0
    for layer in isd_model['layers']:
        layer_name = layer['name']
        layer_class = layer['type']
        if layer_class == 'Conv2D':

            w, d = kreader.get_layer_data(layer_name)
            CK = d[0]
            N = layer['conv_kernel'][2]

            if len(d) > 1:
                bias = d[1]
            else:
                bias = np.zeros(N)

            BK = np.ones(N)

            # if i == 0:
            #     print("BK {} N {} bias {}:".format(CK.shape, N, bias[0]))

            if layer['bn']:
                bnw, bnd = kreader.get_layer_data(layer['bn_layer_name'])
                # Add first Mul later
                BK, B = process_bn(bnd)
                # new bias
                bias += B

            if isd_model['config'].get('input_mode', 'tf') == 'caffe' and is_first_conv:
                print('[INFO] Processing caffe weights')
                BK *= 16

            # MN combine weight
            if mode == 'MN':
                ref_data[layer_name] = BK
                total_size += 4 * CK.size + 4 * BK.size

            # 16 bit quantization
            if mode == 'MNF':
                CK, MX = QuantifyData('SFP6_9', CK)
                BK, MX = QuantifyData('SFP6_9', BK)
                ref_data[layer_name] = BK
                total_size += 2 * CK.size + 2 * BK.size

            if mode in ['MNFIBF']:
                CK, MX = QuantifyData('INT8', CK)
                bias = bias / MX[0]
                q = int(weights_quant[layer_name]['ref'].split('_')[1])
                print(layer_name, 'ref', 'SFP{}_{}'.format(q, 15 - q))
                ref_data[layer_name], c = QuantifyData('SFP{}_{}'.format(q, 15 - q), BK * MX[0])

                total_size += CK.size + BK.size * 2

            kwriter.add_layer_data(layer_name, 0, 'kernel', CK)

            if mode[:3] in ['MNF']:
                # print('**** bias:', weights_quant[layer_name]['bias'])
                q = int(weights_quant[layer_name]['bias'].split('_')[1])
                print(layer_name, 'bias', 'SFP{}_{}'.format(q, 15 - q))
                bias, c = QuantifyData('SFP{}_{}'.format(q, 15 - q), bias)
                total_size += bias.size * 2
            else:
                total_size += bias.size * 4
            kwriter.add_layer_data(layer_name, 1, 'bias', bias)
            is_first_conv = False

        elif layer_class == 'Dense':
            w, d = kreader.get_layer_data(layer_name)
            CK = d[0]
            N = CK.shape[1]

            if mode == 'MNFIBF':
                CK, MX = QuantifyData('INT8', CK)
                q = int(weights_quant[layer_name]['ref'].split('_')[1])
                print(layer_name, 'ref', 'SFP{}_{}'.format(q, 15 - q))
                ref_data[layer_name], c = QuantifyData('SFP{}_{}'.format(q, 15 - q), MX)
                total_size += int(CK.size) + N * 2

            if mode == 'MNF':
                CK, MX = QuantifyData('SFP6_9', CK)

            kwriter.add_layer_data(layer_name, 0, 'kernel', CK)

            if len(d) > 1:
                bias = d[1]
                if mode[:3] in ['MNF']:
                    q = int(weights_quant[layer_name]['bias'].split('_')[1])
                    print(layer_name, 'bias', 'SFP{}_{}'.format(q, 15 - q))
                    bias, c = QuantifyData('SFP{}_{}'.format(q, 15 - q), bias)
                    total_size += N * 2
                else:
                    total_size += N * 4
                kwriter.add_layer_data(layer_name, 1, 'bias', bias)
            is_first_conv = False

        i = i + 1

    print("Save", output_file)
    kwriter.save_hdf5(output_file)
    print("Save", ref_output_file)
    DataSet2H5(ref_data, ref_output_file)
    print("Total Size", total_size)
    print("================ quantization end =======================")


def process_bn(b):
    esp = 0.001
    if len(b) == 3:
        beta = b[0]
        mu = b[1]
        sigma = b[2]
        N = len(beta)
        gamma = np.ones(N)
    elif len(b) == 4:
        gamma = b[0]
        beta = b[1]
        mu = b[2]
        sigma = b[3]
        N = len(beta)

    K = np.empty(N)
    B = np.empty(N)

    for n in range(N):
        if abs(gamma[n]) < 0.01:
            gamma[n] = 0.01
        K[n] = gamma[n] / (math.sqrt(sigma[n] + esp))
        B[n] = (beta[n] - gamma[n] * mu[n] / math.sqrt(sigma[n] + esp)) / K[n]

    return K, B
