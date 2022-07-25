#! /usr/bin/python
# -*- coding: utf-8 -*-

from .common import check_parameter, processing_act, str2init, random_normal, tolist, construct_graph, ModuleNode, select_attrs
from .common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
from mindspore.nn import Cell
import tensorlayerx as tlx
import mindspore as ms
from mindspore import log as logger
import inspect
from mindspore import context
import numpy
from mindspore.common.api import _pynative_executor
from collections import OrderedDict, abc as container_abcs


__all__ = ['Module', 'Sequential', 'ModuleList', 'ModuleDict', 'Parameter', 'ParameterList', 'ParameterDict']

_global_layer_name_dict = {}
_global_layer_node = []


class Module(Cell):

    def __init__(self, name=None, act=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # mindspore auto-naming is set to False
        self._auto_prefix = False
        # Uniform parameter naming
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
        self._all_weights = []
        self._trainable_weights = []
        self._nontrainable_weights = []

        # Layer training state
        self.is_train = True

        # layer forward  state
        self._forward_state = False

        # data_format
        self.data_format = "NCHW"

    def forward(self, *inputs, **kwargs):
        raise Exception("The forward method must be implemented by inherited class")

    def construct(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def build(self, inputs_shape):
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    def _get_weights(self, var_name, shape, init=random_normal(), trainable=True, transposed=False, order=False):
        """ Get trainable variables. """
        var_name = self.name + "/" + var_name
        # TODO 2D mindspore weights shape : [out_channel, in_channel, kernel_h, kernel_w]
        # TODO 2D mindspore transposed shape [in_channel, out_channel, kernel_h, kernel_w]
        if order:
            initial_value = init(shape=shape)
            return tlx.Variable(initial_value=initial_value, name=var_name, trainable=trainable)

        if len(shape) == 3:
            shape = shape[::-1]
        if len(shape) == 4:
            if not transposed and self.data_format in ['NHWC', 'channels_last']:
                shape = (shape[3], shape[0], shape[1], shape[2])
            else:
                shape = (shape[3], shape[2], shape[0], shape[1])
        if len(shape) == 5:
            shape = (shape[4], shape[3], shape[0], shape[1], shape[2])

        initial_value = init(shape=shape)
        var = tlx.Variable(initial_value=initial_value, name=var_name, trainable=trainable)
        self.trainable = trainable
        return var

    def save_weights(self, file_path, format=None):
        """Input file_path, save model weights into a file of given format."""
        _save_weights(self, file_path, format)

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights()."""
        _load_weights(self, file_path, format, in_order, skip)

    def save_standard_weights(self, file_path):
        _save_standard_weights_dict(self, file_path)

    def load_standard_weights(self, file_path, weights_from, weights_to, skip=False):
        _load_standard_weights_dict(self, file_path, skip=skip, weights_from=weights_from, weights_to=weights_to)

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [tlx.get_tensor_shape(t) for t in tensors]
        else:
            shape_mem = tlx.get_tensor_shape(tensors)
        return shape_mem

    # def __call__(self, *args, **kwargs):
    ## TODO With MindSpore __call__, refactoring is required when there are special cases to consider

    def set_train(self):
        """
        Sets the cell to training mode.

        The cell itself and all children cells will be set to training mode.

        Args:
            mode (bool): Specifies whether the model is training. Default: True.
        """
        self._phase = 'train'
        self.add_flags_recursive(training=True)
        return self

    def set_eval(self):
        """Set this network in evaluation mode. After calling this method,
        all layers in network are in evaluation mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayerx as tlx
        >>> net = tlx.model.vgg16()
        >>> net.eval()
        # do evaluation

        """
        self._phase = 'predict'
        self.add_flags_recursive(training=False)
        for layer in self.cells():
            layer.is_train = False
        return self

    def test(self):
        """Set this network in evaluation mode."""
        self.eval()

    def infer(self):
        """Set this network in evaluation mode."""
        self.eval()

    @property
    def trainable_weights(self):
        """
        Returns all trainable weights.

        Returns a list of all trainable parmeters.

        Args:
            recurse (bool): Whether contains the trainable weights of sublayers. Default: True.

        Returns:
            List, the list of trainable weights.
        """
        self._trainable_weights = list(filter(lambda x: x.requires_grad, self.get_parameters(expand=True)))
        return self._trainable_weights

    @property
    def nontrainable_weights(self):
        """
        Returns all untrainable weights.

        Returns a list of all untrainable weights.

        Args:
            recurse (bool): Whether contains the untrainable weights of sublayers. Default: True.

        Returns:
            List, the list of untrainable weights.
        """
        return list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=True)))

    @property
    def all_weights(self):
        return list(filter(lambda x: x.requires_grad, self.get_parameters(expand=True))) \
               + list(filter(lambda x: not x.requires_grad, self.get_parameters(expand=True)))

    def str_to_init(self, initializer):
        return str2init(initializer)

    def check_param(self, param, dim='2d'):
        return check_parameter(param, dim)

    def insert_child_to_layer(self, child_name, child):
        """
        Adds a child layer to the current layer.

        Parameters
        ----------
        child_name : str
            Name of the child layer.
        child : Module
            The child layer to be inserted.

        """

        if not child_name or '.' in child_name:
            raise KeyError("Child layer name is incorrect.")
        if hasattr(self, child_name) and child_name not in self._layers:
            raise KeyError("Duplicate child name '{}'.".format(child_name))
        if not isinstance(child, Module) and child is not None:
            raise TypeError("Child layer type is incorrect.")
        self._cells[child_name] = child

    def init_build(self, *inputs, **kwargs):
        """
        (1) This method must be called when the Layer has no input in_channels.
        (2) Automatic shape inference when the user does not enter in_channels.
        """

        self.forward(*inputs, **kwargs)

    def build_graph(self, *inputs, **kwargs):
        # Add nodes only when the composition is needed.
        for layer_name, layer in self._cells.items():
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
    >>> conv = tlx.nn.Conv2d(3, 2, 3, pad_mode='valid')
    >>> bn = tlx.nn.BatchNorm2d(2)
    >>> seq = tlx.nn.Sequential([conv, bn])
    >>> x = tlx.nn.Input((1, 3, 4, 4))
    >>> seq(x)

    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        # self._built = True
        if len(args) == 1:
            layers = args[0]
            if isinstance(layers, list):
                for index, layer in enumerate(layers):
                    self.insert_child_to_layer(str(index), layer)
            elif isinstance(layers, OrderedDict):
                for name, layer in layers.items():
                    self.insert_child_to_layer(name, layer)
            else:
                raise TypeError('Layers must be list or orderedDict')
        else:
            for index, layer in enumerate(args):
                self.insert_child_to_layer(str(index), layer)
        self.layer_list = list(self._cells.values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(OrderedDict(list(self._cells.items())[index]))
        index = _valid_index(len(self), index)
        return list(self._cells.values())[index]

    def __setitem__(self, index, layer):
        if _valid_module(layer):
            index = _valid_index(len(self), index)
            key = list(self._cells.keys())[index]
            self._cells[key] = layer
            self.layer_list = list(self._cells.values())

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            key = list(self._cells.keys())[index]
            del self._cells[key]
        elif isinstance(index, slice):
            keys = list(self._cells.keys())[index]
            for key in keys:
                del self._cells[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        self.layer_list = list(self._cells.values())

    def __len__(self):
        return len(self._cells)

    def set_grad(self, flag=True):
        self.requires_grad = flag
        for layer in self._cells.values():
            layer.set_grad(flag)

    def append(self, layer):
        if _valid_module(layer):
            self._cells[str(len(self))] = layer
        self.layer_list = list(self._cells.values())
        return self

    def build(self, inputs_shape):
        pass

    def forward(self, input_data):
        for layer in self.layer_list:
            inputs = input_data
            input_data = layer(input_data)
            outputs = input_data
            if not self._nodes_fixed and self._build_graph:
                self._add_seq_node(inputs, outputs, layer)
        self._nodes_fixed = True
        return input_data

    def _add_seq_node(self, input_tensors, output_tensors, layer):
        inputs_list = tolist(input_tensors)
        outputs_list = tolist(output_tensors)
        if layer.__class__.__name__ in tlx.layers.inputs.__all__:
            in_nodes = []
            in_tensor_idxes = [0]
        else:
            in_nodes = [tensor._info[0] for tensor in inputs_list]
            in_tensor_idxes = [tensor._info[1] for tensor in inputs_list]
        node_index = len(_global_layer_node)

        new_node = ModuleNode(
            layer, node_index, in_nodes, inputs_list, outputs_list, in_tensor_idxes, select_attrs(layer)
        )
        _global_layer_node.append(new_node)
        for idx, tensor in enumerate(outputs_list):
            tensor._info = (new_node, idx)


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
    >>> d1 = Linear(out_features=800, act=tlx.ReLU, in_features=784, name='Linear1')
    >>> d2 = Linear(out_features=800, act=tlx.ReLU, in_features=800, name='Linear2')
    >>> d3 = Linear(out_features=10, act=tlx.ReLU, in_features=800, name='Linear3')
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
            return self.__class__(list(self._cells.values())[index])
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            return self._cells[str(index)]
        raise TypeError('Index {} is not int type or slice type'.format(index))

    def __setitem__(self, index, layer):
        if not isinstance(index, int) and _valid_module(layer):
            raise TypeError('Index {} is not int type'.format(index))
        index = _valid_index(len(self), index)
        self._cells[str(index)] = layer

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            del self._cells[str(index)]
        elif isinstance(index, slice):
            keys = list(self._cells.keys())[index]
            for key in keys:
                del self._cells[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        temp_dict = OrderedDict()
        for idx, layer in enumerate(self._cells.values()):
            temp_dict[str(idx)] = layer
        self._cells = temp_dict

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells.values())

    def __iadd__(self, layers):
        self.extend(layers)
        return self

    def insert(self, index, layer):
        idx = _valid_index(len(self), index)
        _valid_module(layer)
        length = len(self)
        while length > idx:
            self._cells[str(length)] = self._cells[str(length - 1)]
            length -= 1
        self._cells[str(idx)] = layer

    def extend(self, layers):

        if not isinstance(layers, list):
            raise TypeError('Modules {} should be list of sublayers'.format(layers))
        for layer in layers:
            if _valid_module(layer):
                self._cells[str(len(self))] = layer
        return self

    def append(self, layer):

        if _valid_module(layer):
            self._cells[str(len(self))] = layer

    def set_grad(self, flag=True):
        self.requires_grad = flag
        for layer in self._cells.values():
            layer.set_grad(flag)

    def forward(self, *inputs):
        raise NotImplementedError


class ModuleDict(Module):

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):

        return self._cells[key]

    def __setitem__(self, key, module):
        if not isinstance(key, str):
            raise TypeError("module name should be a string, but got {}".format(type(key)))
        elif '.' in key:
            raise KeyError("module name can't contain \".\", got: {}".format(key))
        elif key == '':
            raise KeyError("module name can't be empty string \"\"")
        if _valid_module(module):
            self._cells[key] = module

    def __delitem__(self, key):

        del self._cells[key]

    def __len__(self):

        return len(self._cells)

    def __iter__(self):

        return iter(self._cells)

    def __contains__(self, key):

        return key in self._cells

    def clear(self):

        self._cells.clear()

    def pop(self, key):

        temp = self[key]
        del self[key]
        return temp

    def keys(self):

        return self._cells.keys()

    def items(self):

        return self._cells.items()

    def values(self):

        return self._cells.values()

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


def Parameter(data=None, requires_grad=True, name=None):

    return ms.Parameter(default_input=data, requires_grad=requires_grad, name=name)


class ParameterList(Module):

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def _get_abs_string_index(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._params.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._params[str(idx)]

    def __setitem__(self, idx, parameter):
        idx = self._get_abs_string_index(idx)
        self._params[str(idx)] = parameter

    def __setattr__(self, key, value):
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self):
        return len(self._params)

    def __iter__(self):
        return iter(self._params.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter):
        self._params[str(len(self))] = parameter
        return self

    def extend(self, parameters):
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParameterList.extend should be called with an "
                "iterable, but got " + type(parameters).__name__
            )
        offset = len(self)
        for i, para in enumerate(parameters):
            self._params[str(offset + i)] = para
        return self

    def __call__(self, input):
        raise RuntimeError('ParameterList should not be called.')


class ParameterDict(Module):

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, parameter):
        self._params[key] = parameter

    def __delitem__(self, key):
        del self._params[key]

    def __setattr__(self, key, value):
        super(ParameterDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._params)

    def __reversed__(self):
        return reversed(list(self._params.keys()))

    def __iter__(self):
        return iter(self._params.keys())

    def copy(self):
        return ParameterDict(self._params.copy())

    def __contains__(self, key):
        return key in self._params

    def setdefault(self, key, default=None):
        if key in self._params:
            return self._params[key]
        self[key] = default
        return self._params[key]

    def clear(self):
        return self._params.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def popitem(self):
        return self._params.popitem()

    def get(self, key, default=None):
        return self._params.get(key, default)

    def fromkeys(self, keys, default=None):
        return ParameterDict(self._params.fromkeys(keys, default))

    def keys(self):
        return self._params.keys()

    def items(self):
        return self._params.items()

    def values(self):
        return self._params.values()

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
