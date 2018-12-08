# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import absolute_import
import warnings
import h5py
import json
import numpy as np
import argparse
import re
from collections import OrderedDict


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def Element2Symbol(e):
    if e == 0:
        return (0, 0)
    else:
        v = int(e * (1 << 9))
        if v >= 0:
            if v > 0x7FFF:
                v = 0x7FFF
            v = v & 0x0000FFFF
            return (v & 0xFF, (v >> 8) & 0xFF)
        else:
            v = 0 - v
            if v > 0x7FFF:
                v = 0x7FFF
            v = (~v + 1) & 0x00007FFF
            return (v & 0xFF, ((v >> 8) | 0x80) & 0xFF)


def Symbol2Element(e):
    if type(e) == type(''):
        e = int('0x' + e, base=16)
    if e & 0x8000:
        e = (~e + 1) & 0x00007FFF
        e = -1 * e
    return float(e) / 512.0


def DataSet2H5(ds, filename, saveorder=False):
    pf = h5py.File(filename, 'w')

    if type(ds) not in [type({}), type(OrderedDict())]:
        ds = {'_': ds}

    if saveorder:
        keys = ds.keys()
        if '+' not in keys:
            ds['+'] = {}
        else:
            keys.remove('+')
        if '/' not in ds['+'].keys():
            ds['+']['/'] = {}
        ds['+']['/']['__order'] = [str(x).encode('utf-8') for x in list(keys)]

    for k, v in ds.items():
        if k != '+':
            try:
                pf.create_dataset(name=k, data=v)
            except:
                print("[ERROR] invalid type", type(k), type(v))

    if '+' in ds.keys():
        for k, v in ds['+'].items():
            if k == '/':
                ds = pf
            else:
                ds = pf[k]
            for vk, vv in v.items():
                ds.attrs[vk] = vv
    pf.close()


def Dict2JSON(dt, filename):
    fp = open(filename, 'w')
    fp.write(json.dumps(dt, indent=4))
    fp.close()


def __H52DataSet_iter(dt, key, dataset_or_group):
    for a in dataset_or_group.attrs.keys():
        if key not in dt['+'].keys():
            dt['+'][key] = {}
        dt['+'][key][a] = dataset_or_group.attrs[a]
    if str(type(dataset_or_group)) == "<class 'h5py._hl.group.Group'>":
        for item_key in dataset_or_group:
            try:
                __H52DataSet_iter(dt, key + '/' + item_key, dataset_or_group[item_key])
            except:
                print("Fail", item_key)
    else:
        dt[key] = np.array(dataset_or_group)


def H52DataSet(filename):
    pf = h5py.File(filename, 'r')
    dt = {'+': {'/': {}}}

    for a in pf.attrs.keys():
        dt['+']['/'][a] = pf.attrs[a]
    for item in pf:
        __H52DataSet_iter(dt, item, pf[item])
    return dt


def round_to(a, v):
    if a % v == 0:
        return a
    else:
        return (int(a / float(v)) + 1) * v


def GetOpt(app_name, in_list, *args):
    parser = argparse.ArgumentParser(description=app_name)
    type_map = {
        'f': float,
        'i': int,
        's': str
    }
    options = {}
    for arg_item in args:

        if arg_item.startswith('~'):
            op = arg_item.split(':')
            options[op[0]] = op[1]
            continue

        optional_type = ''
        items = [arg_item]

        m = re.match(r'([A-Za-z0-9_]+)([:=])([sif])(.*)', arg_item)
        if m:
            items = [m.group(1), m.group(3), m.group(4)]
            optional_type = m.group(2)

        if len(items) == 1:
            parser.add_argument('-' + arg_item[0], '--' + arg_item, action='store_true', default=False)
        elif len(items) >= 2:
            arg_name = items[0]
            arg_type = items[1]
            arg_default = items[2].replace(':', '')

            if len(arg_default) == 0:
                arg_default = None
            else:
                arg_default = type_map[arg_type](arg_default)
            if arg_type in type_map.keys():
                parser.add_argument('-' + arg_name[0], '--' + arg_name, action='store', required=(optional_type == '='),
                                    type=type_map[arg_type], default=arg_default)

    args = parser.parse_args(in_list)
    rtn = args.__dict__
    if rtn is None and options.get('~no_exit_when_error', 0):
        return None
    return rtn


class KerasWeightReader():
    def __init__(self, model_json_file, model_weight_file):
        self.model_json = json.load(open(model_json_file))
        self.weight_data = H52DataSet(model_weight_file)

        try:
            self.layer_names = [bytes(x).decode('utf-8') for x in self.weight_data['+']['/']['layer_names']]
        except:
            try:
                self.layer_names = [bytes(x).decode('utf-8') for x in
                                    self.weight_data['+']['model_weights']['layer_names']]
            except:
                self.layer_names = [key for key in self.weight_data['+'].keys()]

    def get_layer_data(self, layer_name):

        if layer_name not in self.layer_names:
            print("[ERROR] not found", layer_name)
            return []
        else:
            try:
                layer_attr = self.weight_data['+'][layer_name]
            except:
                layer_attr = self.weight_data['+']['model_weights/' + layer_name]

            weight_names = [bytes(x).decode('utf-8') for x in layer_attr['weight_names']]
            data_set = []
            for w in weight_names:
                data_set.append(self.weight_data[layer_name + '/' + w])
            return weight_names, data_set


class KerasWeightWriter():
    def __init__(self):
        self.layer_names = []
        self.layer_data = {}

    def add_layer_data(self, layer_name, data_id, data_name, data):
        if layer_name not in self.layer_names:
            self.layer_names.append(layer_name)
            self.layer_data[layer_name] = {
                'weight_names': [],
                'data': {
                }
            }
        if data is None:
            return
        weight_names = self.layer_data[layer_name]['weight_names']
        self.layer_data[layer_name]['data'][data_name] = data

        if data_id != -1:
            if data_id >= len(weight_names):
                for i in range(data_id - len(weight_names) + 1):
                    weight_names.append('')
            weight_names[data_id] = 'data/' + data_name

    def save_hdf5(self, file_name):
        data_set = {
            '+': {
                '/': {
                    'backend': b'tensorflow',
                    'keras_version': b'2.0.0-tf',
                    'layer_names': [(str(x).encode('utf-8')) for x in self.layer_names]
                }
            }
        }

        for layer_name in self.layer_data.keys():
            if layer_name not in data_set['+'].keys():
                data_set['+'][layer_name] = {}

            data_set['+'][layer_name]['weight_names'] = [(str(x).encode('utf-8')) for x in
                                                         self.layer_data[layer_name]['weight_names']]
            for k, v in self.layer_data[layer_name]['data'].items():
                data_set[layer_name + '/data/' + k] = v

        DataSet2H5(data_set, file_name)


def QuantifyData(compress_mode, data, layer_type='Default', data_index=-1):
    if compress_mode is None:
        return data, [0]
    m = re.match(r'SFP(\d+)_(\d+)', compress_mode)

    if m:
        int_bits = int(m.group(1))
        decimal_bits = int(m.group(2))
        all_bits = int_bits + decimal_bits
        max_int_v = (1 << all_bits) - 1
        min_int_v = (1 << all_bits) * -1
        max_float_v = float(max_int_v) / float((1 << decimal_bits))
        min_float_v = float(min_int_v) / float((1 << decimal_bits))
        array_data = np.array(data, dtype=np.float) * (1 << decimal_bits)
        array_data = array_data.astype(np.int)
        array_data[array_data > max_int_v] = max_int_v
        array_data[array_data < min_int_v] = min_int_v
        array_data = array_data.astype(np.float) / float((1 << decimal_bits))
        return array_data, [0]

    m = re.match(r'INT(\d)', compress_mode)
    if m:
        int_bits = int(m.group(1))
        max_f = float((1 << (int_bits - 1)))
        l_max = 0
        fpdata = data
        d = np.abs(np.reshape(data, (-1, data.shape[-1])))
        l_max = np.max(d, axis=0)
        l_max[l_max < 1e-3] = 1e-3
        w_revert = np.array(data) * max_f / l_max
        w_revert = np.maximum(w_revert, -1 * max_f)
        w_revert = np.minimum(w_revert, max_f)
        w_int = w_revert.astype(np.int).astype(np.float)
        return w_int / 128.0, [l_max]

    return data, [0]
