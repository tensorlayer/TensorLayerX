#! /usr/bin/python
# -*- coding: utf-8 -*-

from .common import str2act, str2init, random_normal, tolist, construct_graph, ModuleNode
from .common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
from mindspore.nn import Cell
import tensorlayerx as tlx
import mindspore as ms
from mindspore import log as logger
import inspect
from mindspore import context
import numpy
from mindspore.common.api import _pynative_executor
from mindspore.common.parameter import Parameter
from collections import OrderedDict, abc as container_abcs
__all__ = ['Module', 'Sequential', 'ModuleList', 'ModuleDict']

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
            if not transposed and self.data_format == 'NHWC':
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

    def load_standard_weights(self, file_path, skip=False, reshape=False, format='npz_dict'):
        _load_standard_weights_dict(self, file_path, skip, reshape, format)

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [tlx.get_tensor_shape(t) for t in tensors]
        else:
            shape_mem = tlx.get_tensor_shape(tensors)
        return shape_mem

    def __call__(self, *args, **kwargs):
        if self.__class__.construct is Cell.construct:
            logger.warning(f"The '{self.__class__}' does not override the method 'construct', "
                           f"will call the super class(Cell) 'construct'.")
        if kwargs:
            bound_arguments = inspect.signature(self.construct).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            args = bound_arguments.args
            kwargs = bound_arguments.kwargs

        # Run in Graph mode.
        if context._get_mode() == context.GRAPH_MODE:
            self._check_construct_args(*args, **kwargs)
            if self.enable_hook:
                raise ValueError("For 'Cell', it's not support hook function in graph mode, please use "
                                 "context.set_context to set pynative mode.")
            out = self.compile_and_run(*args)
            return out

        # Run in PyNative mode.
        if _pynative_executor.is_top_cell():
            _pynative_executor.set_lazy_build(True)
            # There many Casts in parameter_broadcast. Enable lazy_build and build faster.
            self._do_parameter_broadcast()

        for item in args:
            if isinstance(item, ms.Tensor) and item.has_init:
                item.init_data()
            elif isinstance(item, numpy.ndarray):
                raise TypeError("For 'Cell', inputs should not be numpy array.")
        if self.requires_grad is True:
            _pynative_executor.set_grad_flag(True)
        _pynative_executor.new_graph(self, *args, **kwargs)
        cast_inputs = self.auto_cast_inputs(args)

        with self.CellGuard():
            try:
                output = self.run_construct(cast_inputs, kwargs)
            except Exception as err:
                _pynative_executor.clear_res()
                raise err

        if _pynative_executor.is_top_cell():
            _pynative_executor.execute_lazy_task()

        if isinstance(output, Parameter):
            output = output.data
        _pynative_executor.end_graph(self, output, *args, **kwargs)
        return output

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
        layers = self.cells_and_names(name_prefix='')
        for layer_name, layer in layers:
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

        new_node = ModuleNode(self, node_index, in_nodes, inputs_list, outputs_list, in_tensor_idxes)
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

    def __init__(self, args):
        Module.__init__(self)
        self.extend(args)

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

    def __init__(self, modules):
        super(ModuleDict, self).__init__()
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
