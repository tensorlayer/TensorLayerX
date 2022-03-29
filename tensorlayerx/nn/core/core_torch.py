#! /usr/bin/python
# -*- coding: utf-8 -*-

from torch.nn import Module as T_Module
from .common import str2act, str2init
from .common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
from torch.nn.parameter import Parameter
from collections import OrderedDict
import torch
import operator
from itertools import islice

_global_layer_name_dict = {}

__all__ = ['Module', 'SequentialLayer', 'LayerList']


class Module(T_Module):

    def __init__(self, name=None, act=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global _global_layer_name_dict
        if name is None:
            prefix = self.__class__.__name__.lower()

            if _global_layer_name_dict.get(prefix) is not None:
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
            else:
                _global_layer_name_dict[prefix] = 0
                name = prefix
            while True:
                if _global_layer_name_dict.get(name) is None:
                    break
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
        else:
            if _global_layer_name_dict.get(name) is not None:
                pass
            else:
                _global_layer_name_dict[name] = 0

        self.name = name

        if isinstance(act, str):
            str_act = str2act(act)

        if act:
            if isinstance(act, str) and (len(act) > 5 and act[0:5] == "lrelu" or
                                         len(act) > 10 and act[0:10] == "leaky_relu"):
                self.act = str_act
            elif isinstance(act, str):
                self.act = str_act()
            else:
                self.act = act()
        else:
            self.act = act

        # Layer building state
        self._built = False

        # Layer nodes state
        self._nodes = []
        self._nodes_fixed = False

        # Layer weight state
        self._all_weights = None
        self._trainable_weights = None
        self._nontrainable_weights = None

        # Layer training state
        self.is_train = True

        # layer forward  state
        self._forward_state = False

    def set_train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.is_train = mode
        for module in self.children():
            module.is_train = mode
        return self

    def set_eval(self):
        self.set_train(False)

    def build(self, inputs_shape):
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    def forward(self, *inputs, **kwargs):
        raise Exception("The forward method must be implemented by inherited class")

    def _get_weights(self, var_name, shape, init=None, trainable=True, transposed=None, order=False):
        var_name = self.name + "/" + var_name

        if order:
            w_tmp = Parameter(init(shape), requires_grad=trainable)
            return w_tmp

        if len(shape) == 3:
            shape = shape[::-1]
        if len(shape) == 4:
            if transposed:
                shape = (shape[3], shape[0], shape[1], shape[2])
            else:
                shape = (shape[3], shape[2], shape[0], shape[1])
        if len(shape) == 5:
            shape = (shape[4], shape[3], shape[0], shape[1], shape[2])
        # TODO paramters name should be add
        _param = init(shape)
        param = Parameter(_param, requires_grad=trainable)
        return param

    @property
    def all_weights(self):
        if self._all_weights is not None and len(self._all_weights) > 0:
            # self._all_weights already extracted, so do nothing
            pass
        else:
            self._all_weights = []
            for name, param in self.named_parameters(recurse=True):
                self._all_weights.append(param)
        return self._all_weights

    @property
    def trainable_weights(self):
        if self._trainable_weights is not None and len(self._trainable_weights) > 0:
            # self._trainable_weights already extracted, so do nothing
            pass
        else:
            self._trainable_weights = []
            for name, param in self.named_parameters(recurse=True):
                if param.requires_grad ==True:
                    self._trainable_weights.append(param)
        return self._trainable_weights

    @property
    def nontrainable_weights(self):
        """
        Returns all untrainable weights.
        Returns a list of all untrainable weights.

        """

        if self._nontrainable_weights is not None and len(self._nontrainable_weights) > 0:
            # self._nontrainable_weights already extracted, so do nothing
            pass
        else:
            self._nontrainable_weights = []
            for name, param in self.named_parameters(recurse=True):
                if param.requires_grad == False:
                    self._nontrainable_weights.append(param)
        return self._nontrainable_weights

    def save_weights(self, file_path, format=None):
        _save_weights(net=self, file_path=file_path, format=format)

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights()."""
        _load_weights(net=self, file_path=file_path, format=format, in_order=in_order, skip=skip)

    def save_standard_weights(self, file_path):
        _save_standard_weights_dict(self, file_path)

    def load_standard_weights(self, file_path, skip=False, reshape=False, format='npz_dict'):
        _load_standard_weights_dict(self, file_path, skip, reshape, format)

    def str_to_init(self, initializer):
        return str2init(initializer)

    def init_build(self, *inputs, **kwargs):
        """
        (1) This method must be called when the Layer has no input in_channels.
        (2) Automatic shape inference when the user does not enter inchannels.
        """

        self.forward(*inputs, **kwargs)


class SequentialLayer(Module):
    """
    The class :class:`SequentialLayer` is a linear stack of layers.
    The :class:`SequentialLayer` can be created by passing a list of layer instances.
    The given layer instances will be automatically connected one by one.
    Parameters
    ----------
    layers: list of Layer
        A list of layers.
    name : str or None
        A unique layer name. If None, a unique name will be automatically assigned.
    Methods
    ---------
    __init__()
        Initializing the LayerList.
    weights()
        A collection of weights of all the layer instances.
    build()
        Build the LayerList. The layer instances will be connected automatically one by one.
    forward()
        Forward the computation. The computation will go through all layer instances.

    Examples
    ---------
    >>> conv = tlx.layers.Conv2d(3, 2, 3, pad_mode='valid')
    >>> bn = tlx.layers.BatchNorm2d(2)
    >>> seq = tlx.nn.SequentialLayer([conv, bn])
    >>> x = tlx.layers.Input((1, 3, 4, 4))
    >>> seq(x)
    """

    def __init__(self, *args):
        super(SequentialLayer, self).__init__()
        self._built = True
        if len(args) == 1:
            layers = args[0]
            if isinstance(layers, list):
                for index, layer in enumerate(layers):
                    self.add_module(str(index), layer)
            elif isinstance(layers, OrderedDict):
                for name, layer in layers.items():
                    self.add_module(name, layer)
            else:
                raise TypeError('Layers must be list or orderedDict')
        else:
            for index, layer in enumerate(args):
                self.add_module(str(index), layer)
        self.layer_list = list(self._modules.values())

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        index = _valid_index(len(self), idx)
        return list(self._modules.values())[index]

    def __setitem__(self, index, layer):
        if _valid_module(layer):
            index = _valid_index(len(self), index)
            key = list(self._modules.keys())[index]
            self._modules[key] = layer
            self.layer_list = list(self._modules.values())

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            key = list(self._modules.keys())[index]
            del self._modules[key]
        elif isinstance(index, slice):
            keys = list(self._modules.keys())[index]
            for key in keys:
                del self._modules[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        self.layer_list = list(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(SequentialLayer, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, layer):
        if _valid_module(layer):
            self._modules[str(len(self))] = layer
        self.layer_list = list(self._modules.values())
        return self

    def build(self, inputs_shape):
        pass

    def forward(self, input_data):
        for layer in self.layer_list:
            input_data = layer(input_data)
        return input_data


class LayerList(Module):
    """
    Holds Modules in a list.

    LayerList can be used like a regular Python list, support
    '__getitem__', '__setitem__', '__delitem__', '__len__', '__iter__' and '__iadd__',
    but module it contains are properly registered, and will be visible by all Modules methods.

    Parameters
    ----------
        args : list
            List of subclass of Module.
    Methods
    ---------
    __init__()
        Initializing the Layer.
    insert()
        Inserts a given layer before a given index in the list.
    extend()
        Appends layers from a Python iterable to the end of the list.
    append()
        Appends a given layer to the end of the list.

    Examples
    ---------
    >>> from tensorlayerx.nn import Module, LayerList, Dense
    >>> import tensorlayerx as tlx
    >>> d1 = Dense(n_units=800, act=tlx.ReLU, in_channels=784, name='Dense1')
    >>> d2 = Dense(n_units=800, act=tlx.ReLU, in_channels=800, name='Dense2')
    >>> d3 = Dense(n_units=10, act=tlx.ReLU, in_channels=800, name='Dense3')
    >>> layer_list = LayerList([d1, d2])
    >>> # Inserts a given d2 before a given index in the list
    >>> layer_list.insert(1, d2)
    >>> layer_list.insert(2, d2)
    >>> # Appends d2 from a Python iterable to the end of the list.
    >>> layer_list.extend([d2])
    >>> # Appends a given d3 to the end of the list.
    >>> layer_list.append(d3)
    """

    def __init__(self, args):
        super(LayerList, self).__init__()
        self.extend(args)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(list(self._modules.values())[index])
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            return self._modules[str(index)]
        raise TypeError('Index {} is not int type or slice type'.format(index))

    def __setitem__(self, index, layer):
        if not isinstance(index, int) and _valid_module(layer):
            raise TypeError('Index {} is not int type'.format(index))
        index = _valid_index(len(self), index)
        self._modules[str(index)] = layer

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            del self._modules[str(index)]
        elif isinstance(index, slice):
            keys = list(self._modules.keys())[index]
            for key in keys:
                del self._modules[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        temp_dict = OrderedDict()
        for idx, layer in enumerate(self._modules.values()):
            temp_dict[str(idx)] = layer
        self._modules = temp_dict

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, layers):
        self.extend(layers)
        return self

    def insert(self, index, layer):
        """
            Inserts a given layer before a given index in the list.

        """
        idx = _valid_index(len(self), index)
        _valid_module(layer)
        length = len(self)
        while length > idx:
            self._modules[str(length)] = self._modules[str(length - 1)]
            length -= 1
        self._modules[str(idx)] = layer

    def extend(self, layers):
        """
            Appends layers from a Python iterable to the end of the list.

        """

        if not isinstance(layers, list):
            raise TypeError('Modules {} should be list of sublayers'.format(layers))
        for layer in layers:
            if _valid_module(layer):
                self._modules[str(len(self))] = layer
        return self

    def append(self, layer):
        """
            Appends a given layer to the end of the list.

        """

        if _valid_module(layer):
            self._modules[str(len(self))] = layer

    def forward(self, *inputs):
        raise NotImplementedError


def _valid_index(layer_num, index):
    if not isinstance(index, int):
        raise TypeError("Index {} is not int type")
    if not -layer_num <= index < layer_num:
        raise IndexError("Index should be a number in range [{}, {}), but got {}".format(-layer_num, layer_num, index))
    return index % layer_num


def _valid_module(layer):
    if issubclass(layer.__class__, Module):
        return True
    raise TypeError('Module {} is not subclass of Module'.format(layer))
