#! /usr/bin/python
# -*- coding: utf-8 -*-

from torch.nn import Module as T_Module
from .common import check_parameter, processing_act, str2init, tolist, construct_graph, ModuleNode, select_attrs
from .common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
from torch.nn.parameter import Parameter
from typing import Any, Callable
import torch
import operator
from itertools import islice
from collections import OrderedDict, abc as container_abcs
import warnings
import tensorlayerx as tlx

_global_layer_name_dict = {}
_global_layer_node = []

__all__ = ['Module', 'Sequential', 'ModuleList', 'ModuleDict', 'Parameter', 'ParameterList', 'ParameterDict']


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

        self.act = processing_act(act)

        # Layer building state
        self._built = False

        # Layer nodes state
        self._nodes_fixed = False
        self._build_graph = False

        # Layer weight state
        self._all_weights = None
        self._trainable_weights = None
        self._nontrainable_weights = None

        # Layer training state
        self.is_train = True

        # layer forward  state
        self._forward_state = False

        # weights check state
        self._check = False


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
        self.var_name = var_name
        return param

    def _call_impl_tlx(self, *input, **kwargs):
        if self._check == False:
            _param_name = []
            for name, param in self.named_parameters(recurse=True):
                if name not in _param_name:
                    _param_name.append(name)
                else:
                    raise Exception("parameter name [{}] have be been used. "
                                    "In training, the name of layer can't be same."
                                    "Please check the layers name".format(name))
            self._check = True

        result = self._call_impl(*input, **kwargs)
        return result
    # # TODO RNN enabled after repair
    # __call__: Callable[..., Any] = _call_impl_tlx
    #
    # def _named_members(self, get_members_fn, prefix='', recurse=True):
    #     r"""Helper method for yielding various names + members of modules."""
    #     memo = set()
    #     modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
    #     for module_prefix, module in modules:
    #         members = get_members_fn(module)
    #         for k, v in members:
    #             if v is None or v in memo:
    #                 continue
    #             memo.add(v)
    #             name = module.name + '/' + k
    #             yield name, v

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

    def check_param(self, param, dim='2d'):
        return check_parameter(param, dim)

    def init_build(self, *inputs, **kwargs):
        """
        (1) This method must be called when the Layer has no input in_channels.
        (2) Automatic shape inference when the user does not enter inchannels.
        """

        self.forward(*inputs, **kwargs)

    def build_graph(self, *inputs, **kwargs):
        # Add nodes only when the composition is needed.
        for name, layer in self._modules.items():
            if isinstance(layer, Module):
                layer._build_graph = True
        self.set_eval()

        outputs = self.forward(*inputs, **kwargs)
        self.inputs = inputs
        self.outputs = outputs
        self._node_by_depth, self._all_layers = construct_graph(self.inputs, self.outputs)
        return self._node_by_depth, self._all_layers

    def _add_node(self, input_tensors, output_tensors):
        """Add a ModuleNode for this layer given input_tensors, output_tensors.

        This function should not be called from outside, it should only be called
        in __call__ when building static model.

        Parameters
        ----------
        input_tensors : Tensor or a list of tensors
            Input tensors to this layer.
        output_tensors : Tensor or a list of tensors
            Output tensors to this layer.

        """

        inputs_list = tolist(input_tensors)
        outputs_list = tolist(output_tensors)
        if self.__class__.__name__ in tlx.layers.inputs.__all__:
            # for InputLayer, there should be no in_nodes
            in_nodes = []
            in_tensor_idxes = [0]
        else:
            in_nodes = [tensor._info[0] for tensor in inputs_list]
            in_tensor_idxes = [tensor._info[1] for tensor in inputs_list]
        node_index = len(_global_layer_node)

        new_node = ModuleNode(
            self, node_index, in_nodes, inputs_list, outputs_list, in_tensor_idxes, select_attrs(self)
        )
        _global_layer_node.append(new_node)
        for idx, tensor in enumerate(outputs_list):
            tensor._info = (new_node, idx)


class Sequential(Module):
    """
    The class :class:`Sequential` is a linear stack of layers.
    The :class:`Sequential` can be created by passing a list of layer instances.
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
        Initializing the ModuleList.
    weights()
        A collection of weights of all the layer instances.
    build()
        Build the ModuleList. The layer instances will be connected automatically one by one.
    forward()
        Forward the computation. The computation will go through all layer instances.

    Examples
    ---------
    >>> conv = tlx.layers.Conv2d(3, 2, 3, pad_mode='valid')
    >>> bn = tlx.layers.BatchNorm2d(2)
    >>> seq = tlx.nn.Sequential([conv, bn])
    >>> x = tlx.layers.Input((1, 3, 4, 4))
    >>> seq(x)
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
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
        keys = super(Sequential, self).__dir__()
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


class ModuleList(Module):
    """
    Holds Modules in a list.

    ModuleList can be used like a regular Python list, support
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
    >>> from tensorlayerx.nn import Module, ModuleList, Linear
    >>> import tensorlayerx as tlx
    >>> d1 = Linear(out_features=800, act=tlx.ReLU, in_features=784, name='linear1')
    >>> d2 = Linear(out_features=800, act=tlx.ReLU, in_features=800, name='linear2')
    >>> d3 = Linear(out_features=10, act=tlx.ReLU, in_features=800, name='linear3')
    >>> layer_list = ModuleList([d1, d2])
    >>> # Inserts a given d2 before a given index in the list
    >>> layer_list.insert(1, d2)
    >>> layer_list.insert(2, d2)
    >>> # Appends d2 from a Python iterable to the end of the list.
    >>> layer_list.extend([d2])
    >>> # Appends a given d3 to the end of the list.
    >>> layer_list.append(d3)
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self.extend(modules)

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


class ModuleDict(Module):

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):

        return self._modules[key]

    def __setitem__(self, key, module):

        self.add_module(key, module)

    def __delitem__(self, key):

        del self._modules[key]

    def __len__(self):

        return len(self._modules)

    def __iter__(self):

        return iter(self._modules)

    def __contains__(self, key):

        return key in self._modules

    def clear(self):

        self._modules.clear()

    def pop(self, key):

        v = self[key]
        del self[key]
        return v

    def keys(self):

        return self._modules.keys()

    def items(self):

        return self._modules.items()

    def values(self):

        return self._modules.values()

    def update(self, modules):
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(modules).__name__
            )

        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                self[m[0]] = m[1]


class ParameterList(Module):

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        self._initialized = True
        if parameters is not None:
            self += parameters

    def __setstate__(self, state):
        state['_initialized'] = False
        super(ParameterList, self).__setstate__(state)
        self._initialized = True

    def _get_abs_string_index(self, idx):
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._parameters.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        idx = self._get_abs_string_index(idx)
        return self.register_parameter(str(idx), param)

    def __setattr__(self, key, value):
        if getattr(self, "_initialized", False):
            if not hasattr(self, key) and not isinstance(value, torch.nn.Parameter):
                warnings.warn("Setting attributes on ParameterList is not supported.")
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter: 'Parameter') -> 'ParameterList':

        self.register_parameter(str(len(self)), parameter)
        return self

    def extend(self, parameters):
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParameterList.extend should be called with an "
                "iterable, but got " + type(parameters).__name__
            )
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self

    def __call__(self, input):
        raise RuntimeError('ParameterList should not be called.')


class ParameterDict(Module):

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        self._initialized = True
        if parameters is not None:
            self.update(parameters)

    def __setstate__(self, state):
        state['_initialized'] = False
        super(ParameterDict, self).__setstate__(state)
        self._initialized = True

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, parameter):
        self.register_parameter(key, parameter)

    def __delitem__(self, key):
        del self._parameters[key]

    def __setattr__(self, key, value):
        if getattr(self, "_initialized", False):
            if not hasattr(self, key) and not isinstance(value, torch.nn.Parameter):
                warnings.warn("Setting attributes on ParameterDict is not supported.")
        super(ParameterDict, self).__setattr__(key, value)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.keys())

    def __reversed__(self):
        return reversed(list(self._parameters.keys()))

    def copy(self):

        return ParameterDict(self._parameters.copy())

    def __contains__(self, key):
        return key in self._parameters

    def setdefault(self, key, default=None):
        if key in self._parameters:
            return self._parameters[key]
        self[key] = default
        return self._parameters[key]

    def clear(self):
        self._parameters.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def popitem(self):
        return self._parameters.popitem()

    def get(self, key, default=None):

        return self._parameters.get(key, default)

    def fromkeys(self, keys, default=None):

        return ParameterDict(self._parameters.fromkeys(keys, default))  # type: ignore[arg-type]

    def keys(self):

        return self._parameters.keys()

    def items(self):

        return self._parameters.items()

    def values(self):

        return self._parameters.values()

    def update(self, parameters):
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParametersDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(parameters).__name__
            )

        if isinstance(parameters, (OrderedDict, ParameterDict)):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, container_abcs.Mapping):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, container_abcs.Iterable):
                    raise TypeError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(p).__name__
                    )
                print(p)
                if not len(p) == 2:
                    raise ValueError(
                        "ParameterDict update sequence element "
                        "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                    )
                # parameters as length-2 list too cumbersome to type, see ModuleDict.update comment
                self[p[0]] = p[1]  # type: ignore[assignment]

    def __call__(self, input):
        raise RuntimeError('ParameterDict should not be called.')


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
