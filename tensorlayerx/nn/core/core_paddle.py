#! /usr/bin/python
# -*- coding: utf-8 -*-

import copy, six

import paddle

from .common import check_parameter, processing_act, str2init
from .common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
from paddle.fluid import framework
from paddle.fluid.dygraph import Layer
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.dygraph.base import program_desc_tracing_guard, param_guard
from paddle.fluid.dygraph import parallel_helper
import paddle as pd
from collections import OrderedDict, abc as container_abcs

_global_layer_name_dict = {}
_global_layer_node = []

__all__ = ['Module', 'Sequential', 'ModuleList', 'ModuleDict', 'Parameter', 'ParameterList', 'ParameterDict']


class Module(Layer):

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
        # paddl_built
        self._paddle_built = False

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

    def set_train(self):
        """
        Sets this Layer and all its sublayers to training mode.
        This only effects certain modules like `Dropout` and `BatchNorm`.

        Returns:
            None

        Example::
            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self._linear = paddle.nn.Linear(1, 1)
                        self._dropout = paddle.nn.Dropout(p=0.5)

                    def forward(self, input):
                        temp = self._linear(input)
                        temp = self._dropout(temp)
                        return temp

                x = paddle.randn([10, 1], 'float32')
                mylayer = MyLayer()
                mylayer.eval()  # set mylayer._dropout to eval mode
                out = mylayer(x)
                mylayer.train()  # set mylayer._dropout to train mode
                out = mylayer(x)

        """
        # global setting in dygraph
        # NOTE(chenweihang): nn.Layer also can be used in static mode,
        # but _dygraph_tracer() can not be called in static mode
        if in_dygraph_mode():
            framework._dygraph_tracer().train_mode()
        # Layer-level setting
        self.is_train = True
        for layer in self.sublayers():
            layer.is_train = True

    def set_eval(self):
        """
        Sets this Layer and all its sublayers to evaluation mode.
        This only effects certain modules like `Dropout` and `BatchNorm`.

        Returns:
            None

        Example::
            .. code-block:: python

                import paddle

                class MyLayer(paddle.nn.Layer):
                    def __init__(self):
                        super(MyLayer, self).__init__()
                        self._linear = paddle.nn.Linear(1, 1)
                        self._dropout = paddle.nn.Dropout(p=0.5)

                    def forward(self, input):
                        temp = self._linear(input)
                        temp = self._dropout(temp)
                        return temp

                x = paddle.randn([10, 1], 'float32')
                mylayer = MyLayer()
                mylayer.eval()  # set mylayer._dropout to eval mode
                out = mylayer(x)
                print(out)

        """
        # global setting in dygraph
        # NOTE(chenweihang): nn.Layer also can be used in static mode,
        # but _dygraph_tracer() can not be called in static mode
        if in_dygraph_mode():
            framework._dygraph_tracer().eval_mode()
        # Layer-level setting
        self.is_train = False
        for layer in self.sublayers():
            layer.is_train = False

    def build(self, inputs_shape):
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    def forward(self, *inputs, **kwargs):
        raise Exception("The forward method must be implemented by inherited class")

    def __call__(self, *inputs, **kwargs):
        with param_guard(self._parameters), param_guard(self._buffers):
            for forward_pre_hook in self._forward_pre_hooks.values():
                hook_result = forward_pre_hook(self, inputs)
                if hook_result is not None:
                    if not isinstance(hook_result, tuple):
                        hook_result = (hook_result, )
                    inputs = hook_result

            if not self._paddle_built:
                with program_desc_tracing_guard(False):
                    self._build_once(*inputs, **kwargs)
                    if parallel_helper._is_data_parallel_mode():
                        parallel_helper._broadcast_parameters(self._parameters.values())
                self._paddle_built = True

            outputs = self.forward(*inputs, **kwargs)

            for forward_post_hook in self._forward_post_hooks.values():
                hook_result = forward_post_hook(self, inputs, outputs)
                if hook_result is not None:
                    outputs = hook_result

            return outputs

    def _get_weights(self, var_name, shape, init=None, trainable=True, transposed=None, order=False):
        # TODO 2D paddlepaddle weights shape : [out_channel, in_channel, kernel_h, kernel_w]
        # TODO 2D paddlepaddle transposed shape [in_channel, out_channel, kernel_h, kernel_w]
        var_name = self.name + "/" + var_name

        if order:
            w_tmp = self.create_parameter(shape=shape, attr=init, var_name=var_name, trainable=trainable)
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

        w_tmp = self.create_parameter(shape=shape, attr=init, var_name=var_name, trainable=trainable)

        return w_tmp

    def create_parameter(
        self, shape, attr=None, dtype=None, is_bias=False, default_initializer=None, trainable=True, var_name=None
    ):
        """Create parameters for this layer."""
        init_attr = pd.ParamAttr(name=var_name, initializer=attr, trainable=trainable, do_model_average=True)
        temp_attr = copy.deepcopy(init_attr)
        if isinstance(temp_attr, six.string_types) and temp_attr == "":
            temp_attr = None
        return self._helper.create_parameter(temp_attr, shape, dtype, is_bias, default_initializer)

    @property
    def all_weights(self):
        return self.parameters()

    @property
    def trainable_weights(self):
        if self._trainable_weights is not None and len(self._trainable_weights) > 0:
            # self._trainable_weights already extracted, so do nothing
            pass
        else:
            self._trainable_weights = []
            for param in self.parameters():
                if not param.stop_gradient:
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
            for param in self.parameters():
                if param.stop_gradient:
                    self._nontrainable_weights.append(param)
        return self._nontrainable_weights

    def init_build(self, *inputs, **kwargs):
        """
        (1) This method must be called when the Layer has no input in_channels.
        (2) Automatic shape inference when the user does not enter inchannels.
        """

        self.forward(*inputs, **kwargs)

    def save_weights(self, file_path, format=None):
        _save_weights(net=self, file_path=file_path, format=format)

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights()."""
        _load_weights(net=self, file_path=file_path, format=format, in_order=in_order, skip=skip)

    def save_standard_weights(self, file_path):
        _save_standard_weights_dict(self, file_path)

    def load_standard_weights(self, file_path, weights_from, weights_to, skip=False):
        _load_standard_weights_dict(self, file_path, skip=skip, weights_from=weights_from, weights_to=weights_to)

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
        if hasattr(self, child_name) and child_name not in self._sub_layers:
            raise KeyError("Duplicate child name '{}'.".format(child_name))
        if not isinstance(child, Module) and child is not None:
            raise TypeError("Child layer type is incorrect.")
        self._sub_layers[child_name] = child

    def set_build_graph(self):

        raise NotImplementedError

    def build_graph(self, *inputs, **kwargs):
        # Add nodes only when the composition is needed.
        # for layer in self.sublayers():
        #     if isinstance(layer, Module):
        #         layer._build_graph = True
        #
        # outputs = self.forward(*inputs, **kwargs)
        # self.inputs = inputs
        # self.outputs = outputs
        # self._node_by_depth, self._all_layers = construct_graph(self.inputs, self.outputs)
        # return self._node_by_depth, self._all_layers
        raise NotImplementedError

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
        pass
        # inputs_list = tolist(input_tensors)
        # outputs_list = tolist(output_tensors)
        # if self.__class__.__name__ in tlx.layers.inputs.__all__:
        #     # for InputLayer, there should be no in_nodes
        #     in_nodes = []
        #     in_tensor_idxes = [0]
        # else:
        #     in_nodes = [tensor[0] for tensor in inputs_list]
        #     in_tensor_idxes = [tensor[1] for tensor in inputs_list]
        # node_index = len(_global_layer_node)
        #
        # new_node = ModuleNode(self, node_index, in_nodes, inputs_list, outputs_list, in_tensor_idxes)
        # _global_layer_node.append(new_node)
        # for idx, tensor in enumerate(outputs_list):
        #     tensor._info = (new_node, idx)


class Sequential(Module):

    def __init__(self, *args):
        super(Sequential, self).__init__()
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
        self.layer_list = list(self._sub_layers.values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(OrderedDict(list(self._sub_layers.items())[index]))
        index = _valid_index(len(self), index)
        return list(self._sub_layers.values())[index]

    def __setitem__(self, index, layer):
        index = _valid_index(len(self), index)
        key = list(self._sub_layers.keys())[index]
        self._sub_layers[key] = layer
        self.layer_list = list(self._sub_layers.values())

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            key = list(self._sub_layers.keys())[index]
            del self._sub_layers[key]
        elif isinstance(index, slice):
            keys = list(self._sub_layers.keys())[index]
            for key in keys:
                del self._sub_layers[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        self.layer_list = list(self._sub_layers.values())

    def __len__(self):
        return len(self._sub_layers)

    def append(self, layer):

        if _valid_module(layer):
            self._sub_layers[str(len(self))] = layer
        self.layer_list = list(self._sub_layers.values())
        return self

    def forward(self, input_data):
        for layer in self.layer_list:
            input_data = layer(input_data)
        return input_data


class ModuleList(Module):

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self.extend(modules)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(list(self._sub_layers.values())[index])
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            return self._sub_layers[str(index)]
        raise TypeError('Index {} is not int type or slice type'.format(index))

    def __setitem__(self, index, layer):
        if not isinstance(index, int) and _valid_module(layer):
            raise TypeError('Index {} is not int type'.format(index))
        index = _valid_index(len(self), index)
        self._sub_layers[str(index)] = layer

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            del self._sub_layers[str(index)]
        elif isinstance(index, slice):
            keys = list(self._sub_layers.keys())[index]
            for key in keys:
                del self._sub_layers[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        temp_dict = OrderedDict()
        for idx, layer in enumerate(self._sub_layers.values()):
            temp_dict[str(idx)] = layer
        self._sub_layers = temp_dict

    def __len__(self):
        return len(self._sub_layers)

    def __iter__(self):
        return iter(self._sub_layers.values())

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
            self._sub_layers[str(length)] = self._sub_layers[str(length - 1)]
            length -= 1
        self._sub_layers[str(idx)] = layer

    def extend(self, layers):
        """
            Appends layers from a Python iterable to the end of the list.

        """

        if not isinstance(layers, list):
            raise TypeError('Modules {} should be list of sublayers'.format(layers))
        for layer in layers:
            if _valid_module(layer):
                self._sub_layers[str(len(self))] = layer
        return self

    def append(self, layer):
        """
            Appends a given layer to the end of the list.

        """

        if _valid_module(layer):
            self._sub_layers[str(len(self))] = layer

    def forward(self, *inputs):
        raise NotImplementedError


class ModuleDict(Module):

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):

        return self._sub_layers[key]

    def __setitem__(self, key, module):
        if not isinstance(key, str):
            raise TypeError("module name should be a string, but got {}".format(type(key)))
        elif '.' in key:
            raise KeyError("module name can't contain \".\", got: {}".format(key))
        elif key == '':
            raise KeyError("module name can't be empty string \"\"")
        if _valid_module(module):
            self._sub_layers[key] = module

    def __delitem__(self, key):

        del self._sub_layers[key]

    def __len__(self):

        return len(self._sub_layers)

    def __iter__(self):

        return iter(self._sub_layers)

    def __contains__(self, key):

        return key in self._sub_layers

    def clear(self):

        self._sub_layers.clear()

    def pop(self, key):

        temp = self[key]
        del self[key]
        return temp

    def keys(self):

        return self._sub_layers.keys()

    def items(self):

        return self._sub_layers.items()

    def values(self):

        return self._sub_layers.values()

    def update(self, modules):

        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleDict.update should be called with an "
                "iterable of key/value pairs, but got " + type(modules)
            )
        if isinstance(modules, (OrderedDict, ModuleDict, container_abcs.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iterable; is" + type(m)
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                self[m[0]] = m[1]


def Parameter(data=None, requires_grad=True, name=None):
    if data is None:
        data = paddle.empty(0)
    shape = data.shape
    dtype = data.dtype
    parameter = paddle.create_parameter(shape=shape, dtype=dtype)
    parameter.set_value(data)
    return parameter


class ParameterList(Module):

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            for idx, param in enumerate(parameters):
                self.add_parameter(str(idx), param)

    def _get_abs_string_index(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        with param_guard(self._parameters):
            return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        setattr(self, str(idx), param)

    def __setattr__(self, key, value):
        super(ParameterList, self).__setattr__(key, value)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        with param_guard(self._parameters):
            return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter):
        idx = len(self._parameters)
        self.add_parameter(str(idx), parameter)
        return self

    def extend(self, parameters):
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParameterList.extend should be called with an "
                "iterable, but got " + type(parameters).__name__
            )
        offset = len(self)
        for i, para in enumerate(parameters):
            self.add_parameter(str(offset + i), para)
        return self

    def __call__(self, input):
        raise RuntimeError('ParameterList should not be called.')


class ParameterDict(Module):

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, parameter):
        self._parameters[key] = parameter

    def __delitem__(self, key):
        del self._parameters[key]

    def __setattr__(self, key, value):
        super(ParameterDict, self).__setattr__(key, value)

    def __len__(self) -> int:
        return len(self._parameters)

    def __reversed__(self):
        return reversed(list(self._parameters.keys()))

    def __iter__(self):
        return iter(self._parameters.keys())

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
        return self._parameters.clear()

    def pop(self, key):
        v = self[key]
        del self[key]
        return v

    def popitem(self):
        return self._parameters.popitem()

    def get(self, key, default=None):
        return self._parameters.get(key, default)

    def fromkeys(self, keys, default=None):
        return ParameterDict(self._parameters.fromkeys(keys, default))

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
