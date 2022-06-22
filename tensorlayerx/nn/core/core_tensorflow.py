#! /usr/bin/python
# -*- coding: utf-8 -*-

from .common import check_parameter, processing_act, str2init, tolist, construct_graph, ModuleNode, select_attrs
from .common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
from collections import OrderedDict, abc as container_abcs
import warnings
import time
import tensorlayerx as tlx
import tensorflow as tf
from tensorlayerx.nn.layers.utils import (get_variable_with_initializer, random_normal)

__all__ = ['Module', 'Sequential', 'ModuleList', 'ModuleDict', 'Parameter', 'ParameterList', 'ParameterDict']

_global_layer_name_dict = {}
_global_layer_node = []


class Module(object):
    """The basic :class:`Module` class represents a single layer of a neural network.
        It should be subclassed when implementing new types of layers.
        Parameters
        ----------
        name : str or None
            A unique layer name. If None, a unique name will be automatically assigned.
        Methods
        ---------
        __init__()
            Initializing the Layer.
        __call__()
            Forwarding the computation.
        all_weights()
            Return a list of Tensor which are all weights of this Layer.
        trainable_weights()
            Return a list of Tensor which are all trainable weights of this Layer.
        nontrainable_weights()
            Return a list of Tensor which are all nontrainable weights of this Layer.
        build()
            Abstract method. Build the Layer. All trainable weights should be defined in this function.
        _get_weights()
            Abstract method.Create weights for training parameters.
        save_weights()
            Input file_path, save model weights into a file of given format.
        load_weights()
            Load model weights from a given file, which should be previously saved by self.save_weights().
        save_standard_weights()
            Input file_path, save model weights into a npz_dict file. These parameters can support multiple backends.
        load_standard_weights()
            Load model weights from a given file, which should be previously saved by self.save_standard_weights().
        forward()
            Abstract method. Forward computation and return computation results.

        """

    def __init__(self, name=None, act=None, *args, **kwargs):
        self._params = OrderedDict()
        self._layers = OrderedDict()
        # self._params_list = OrderedDict()
        # self._params_dict = OrderedDict()
        self._params_status = OrderedDict()
        self._parameter_layout_dict = {}
        self._create_time = int(time.time() * 1e9)

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

        # layer forward  state
        self._forward_state = False

        # Layer training state
        self.is_train = True

        # weights check state
        self._check = False
        self.trainable = True

    def extend_repr(self):
        """
        Sets the extended representation of the Module.

        To print customized extended information, re-implement this method in your own Layers.

        """

        return ''

    def __repr__(self):
        extra_str = self.extend_repr()
        info_str = self.__class__.__name__ + '<'
        if self._layers:
            sub_str = '\n'
            if extra_str:
                sub_str += '{}\n'.format(self.extend_repr())
            for key, value in self._layers.items():
                sub_str += '({}): {}\n'.format(key, repr(value))
            sub_str = sub_str.replace('\n', '\n  ') + '>'
            info_str += sub_str
        else:
            info_str += extra_str + '>'
        return info_str

    def __setattr__(self, name, value):
        layers = self.__dict__.get('_layers')
        params = self.__dict__.get('_params')

        if isinstance(value, tf.Variable):
            if params is None:
                raise AttributeError("Can not assign params before Module.__init__() call.")
            if name in self.__dict__:
                if self.__dict__[name] is not None:
                    raise TypeError("Expected type is not in (Parameter, Module), but got Parameter.")
                del self.__dict__[name]
            if layers and name in layers:
                raise TypeError("Expected type is Module, but got Parameter.")
            self.insert_param_to_layer(name, value)

        # elif isinstance(value, ParameterList):
        #     self.set_attr_for_parameter_tuple(name, value)
        #
        # elif isinstance(value, ParameterDict):
        #     self.set_attr_for_parameter_dict(name, value)

        elif isinstance(value, Module):
            if layers is None:
                raise AttributeError("Can not assign layers before Module.__init__() call.")
            if name in self.__dict__:
                del self.__dict__[name]
            if params and name in params:
                raise TypeError("Expected type is Parameter, but got Module.")
            layers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __call__(self, inputs, *args, **kwargs):
        if self._check == False:
            self.train_weights_check()
            self._check = True

        output = self.forward(inputs, *args, **kwargs)
        return output

    def train_weights_check(self):
        _param_name = []
        for w in self.trainable_weights:
            if w.name not in _param_name:
                _param_name.append(w.name)
            else:
                raise Exception("parameter name [{}] have be been used. "
                "In training, the name of layer can't be same."
                "Please check the layers name".format(w.name))

    def forward(self, *inputs, **kwargs):
        raise Exception("The forward method must be implemented by inherited class")

    def build(self, inputs_shape):
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    def _get_weights(self, var_name, shape, init=random_normal(), trainable=True, transposed=None, order=False):
        """ Get trainable variables. """

        weight = get_variable_with_initializer(
            scope_name=self.name, var_name=var_name, shape=shape, init=init, trainable=trainable
        )
        self.trainable = trainable
        return weight

    def save_weights(self, file_path, format=None):
        """Input file_path, save model weights into a file of given format."""

        _save_weights(self, file_path, format)

    def load_weights(self, file_path, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights()."""

        _load_weights(self, file_path, format, in_order, skip)

    def save_standard_weights(self, file_path):
        """Save to standard format parameter format as {conv.filters: filters_param, conv.biases: biases_parm,
        linear.weights: weights_parm ...}

        Parameters
        ----------
        file_path : str
            Name of the saved file

        """

        _save_standard_weights_dict(self, file_path)

    def load_standard_weights(self, file_path, skip=False, reshape=False, format='npz_dict'):
        """

        Parameters
        ----------
        file_path : str
            Name of the saved file.
        skip : boolean
            If 'skip' == True, loaded layer whose name is not found in 'layers' will be skipped. If 'skip' is False,
            error will be raised when mismatch is found. Default False.
        reshape : boolean
            This parameter needs to be set to True when importing parameters from tensorflow training to paddle/mindspore/pytorch,
            and similarly when importing parameters from paddle/mindspore/pytorch training to tensorflow.
            This parameter does not need to be set between paddle/mindspore/pytorch.

        """

        _load_standard_weights_dict(self, file_path, skip, reshape, format)

    def _set_mode_for_layers(self, is_train):
        """Set all layers of this network to a given mode.

        Parameters
        ----------
        is_train : boolean
            Network's mode. True means training mode while False means evaluation mode.

        """

        layers = self.layers_and_names(name_prefix='')
        for layer_name, layer in layers:
            if isinstance(layer, Module):
                layer.is_train = is_train

    # def set_attr_for_parameter_dict(self, name, value):
    #     """Set attr for parameter in ParameterDict."""
    #     params = self.__dict__.get('_params')
    #     params_dict = self.__dict__.get('_params_dict')
    #     if params is None:
    #         raise AttributeError("For 'Module', can not assign params before Module.__init__() is called.")
    #     exist_names = set("")
    #     for item in value:
    #         self.insert_param_to_layer(item, value[item], check_name=False)
    #         if item in exist_names:
    #             raise ValueError("The value {} , its name '{}' already exists.".
    #                              format(value[item], item))
    #         exist_names.add(item)
    #
    #     if name in self.__dict__:
    #         del self.__dict__[name]
    #     if name in params:
    #         del params[name]
    #     params_dict[name] = value
    #
    # def set_attr_for_parameter_tuple(self, name, value):
    #     """Set attr for parameter in ParameterTuple."""
    #     params = self.__dict__.get('_params')
    #     params_list = self.__dict__.get('_params_list')
    #     if params is None:
    #         raise AttributeError("For 'Module', can not assign params before Module.__init__() is called.")
    #     exist_names = set("")
    #
    #     for item in value:
    #         self.insert_param_to_layer(item.name, item, check_name=False)
    #         if item.name in exist_names:
    #             raise ValueError("The value {} , its name '{}' already exists.".
    #                              format(value, item.name))
    #         exist_names.add(item.name)
    #
    #     if name in self.__dict__:
    #         del self.__dict__[name]
    #     if name in params:
    #         del params[name]
    #     params_list[name] = value

    def set_train(self):
        """Set this network in training mode. After calling this method,
        all layers in network are in training mode, in particular, BatchNorm, Dropout, etc.
        TODO It is not possible to modify the parameter state after initialization, and a better way needs to be found.
        Examples
        --------
        >>> import tensorlayerx as tlx
        >>> net = tlx.vgg16()
        >>> net.set_train()

        """

        if self.is_train !=True:
            self.is_train = True
            self._set_mode_for_layers(True)

    def set_eval(self):
        """Set this network in evaluation mode. After calling this method,
        all layers in network are in evaluation mode, in particular, BatchNorm, Dropout, etc.
        TODO It is not possible to modify the parameter state after initialization, and a better way needs to be found.
        Examples
        --------
        >>> import tensorlayerx as tlx
        >>> net = tlx.vgg16()
        >>> net.set_eval()
        # do evaluation

        """

        if self.is_train != False:
            self.is_train = False
            self._set_mode_for_layers(False)

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [tlx.get_tensor_shape(t) for t in tensors]
        else:
            shape_mem = tlx.get_tensor_shape(tensors)
        return shape_mem

    def insert_param_to_layer(self, param_name, param, check_name=True):
        """
        Adds a parameter to the current layer.

        Inserts a parameter with given name to the layer. Please refer to the usage in
        source code of `tensorlayerx.layer.Module.__setattr__`.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        param : Parameter
            Parameter to be inserted to the layer.
        check_name : bool
            Determines whether the name input is compatible. Default: True.

        """
        if not param_name:
            raise KeyError("The name of parameter should not be null.")
        if check_name and '.' in param_name:
            raise KeyError("The name of parameter should not contain \".\"")
        if '_params' not in self.__dict__:
            raise AttributeError("You need call init() first.")
        if hasattr(self, param_name) and param_name not in self._params:
            raise KeyError("Duplicated parameter name '{}'.".format(param_name))
        if not isinstance(param, tf.Variable) and param is not None:
            raise TypeError("The type of parameter should be 'Parameter' if not None.")
        self._params[param_name] = param
        try:
            self._params_status[param_name] = self.trainable
        except:
            pass

    @property
    def create_time(self):
        return self._create_time

    def __getattr__(self, name):
        if '_params' in self.__dict__:
            params = self.__dict__['_params']
            if name in params:
                return params[name]
        if '_layers' in self.__dict__:
            layers = self.__dict__['_layers']
            if name in layers:
                return layers[name]
        if '_params_status' in self.__dict__:
            params_status = self.__dict__['_params_status']
            if name in params_status:
                return params_status[name]
        # if '_params_list' in self.__dict__:
        #     params_list = self.__dict__['_params_list']
        #     if name in params_list:
        #         para_list = params_list[name]
        #         return para_list
        # if '_params_dict' in self.__dict__:
        #     params_dict = self.__dict__['_params_dict']
        #     if name in params_dict:
        #         return params_dict[name]
        raise AttributeError("'{}' object has no attribute '{}'.".format(type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._params:
            del self._params[name]
        elif name in self._layers:
            del self._layers[name]
        else:
            object.__delattr__(self, name)

    @property
    def trainable_weights(self):
        """
        Returns all trainable weights.
        Returns a list of all trainable parmeters.

        """

        if self._trainable_weights is not None and len(self._trainable_weights) > 0:
            # self._trainable_weights already extracted, so do nothing
            pass
        else:
            self._trainable_weights = []
            layers = self.layers_and_names(name_prefix='')
            for layer_name, layer in layers:
                params = layer._params.items()
                params_status = layer._params_status.items()
                params_zip = zip(params, params_status)
                for params, params_status in params_zip:
                    if params_status[1] ==True:
                        self._trainable_weights.append(params[1])
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
            layers = self.layers_and_names(name_prefix='')
            for layer_name, layer in layers:
                params = layer._params.items()
                params_status = layer._params_status.items()
                params_zip = zip(params, params_status)
                for params, params_status in params_zip:
                    if params_status[1] == False:
                        self._nontrainable_weights.append(params[1])
        return self._nontrainable_weights

    @property
    def all_weights(self):
        """
        Returns all weights.
        Returns a list of all weights.

        """

        if self._all_weights is not None and len(self._all_weights) > 0:
            # self._all_weights already extracted, so do nothing
            pass
        else:
            self._all_weights = []
            layers = self.layers_and_names(name_prefix='')
            for layer_name, layer in layers:
                params = layer._params.items()
                for par, val in params:
                    self._all_weights.append(val)
        return self._all_weights

    def get_weights(self, expand=True):
        """
        Returns an iterator over layer weights.
        Yields weights of this layer. If `expand` is True, yield parameters of this layer and all sublayers.

        Parameters
        ----------
        expand : bool
            If True, yields parameters of this layer and all sublayers. Otherwise, yields only parameters
            that are direct members of this layer. Default: True.

        Examples
        ---------
        >>> net = Net()
        >>> for item in net.get_weights():
        >>>     print(item)

        """

        for _, param in self.parameters_and_names(expand=expand):
            yield param

    def check_names(self):
        names = set("")
        for value, param in self.parameters_and_names():
            if param.name in names:
                raise ValueError(
                    "The value of {} is {}, its name '{}' already exists.".format(value, param, param.name)
                )
            names.add(param.name)

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
        self._layers[child_name] = child

    def parameters_and_names(self, name_prefix='', expand=True):
        """
        Returns an iterator over layer parameters.

        Includes the parameter's name  and itself.

        Parameters
        ----------
        name_prefix : str
            Namespace. Default: ''.
        expand : bool
            If True, yields parameters of this layer and all sublayers. Otherwise, yields only parameters
            that are direct members of this layer. Default: True.

        Examples
        ---------
        >>> n = Net()
        >>> names = []
        >>> for m in n.parameters_and_names():
        >>>     if m[0]:
        >>>         names.append(m[0])

        """

        layers = []
        if expand:
            layers = self.layers_and_names(name_prefix=name_prefix)
        else:
            layers.append((name_prefix, self))

        params_set = set()
        for layer_name, layer in layers:
            params = layer._params.items()
            for par_name, par in params:
                if par.inited_param is not None:
                    par = par.inited_param
                if par is not None and id(par) not in params_set:
                    params_set.add(id(par))
                    par_new_name = par_name
                    if layer_name:
                        par_new_name = layer_name + '.' + par_new_name

                    yield par_new_name, par

    def layers_and_names(self, layers=None, name_prefix=''):
        """
        Returns an iterator over all layers in the network.

        Includes the layer's name and itself.

        Parameters
        ----------
        layers : str
            layers to iterate over. Default: None.
        name_prefix : str
            Namespace. Default: ''.

        Examples
        ---------
        >>> n = Net()
        >>> names = []
        >>> for m in n.layers_and_names():
        >>>     if m[0]:
        >>>         names.append(m[0])

        """

        t_layers = layers if layers else set()
        if self in t_layers:
            return

        t_layers.add(self)
        yield name_prefix, self

        for name, layer in self._layers.items():
            if layer:
                layers_name_prefix = name
                if name_prefix:
                    layers_name_prefix = name_prefix + '.' + layers_name_prefix
                for ele in layer.layers_and_names(t_layers, layers_name_prefix):
                    yield ele

    def layers(self):
        """Returns an iterator over immediate layers."""

        return self.name_layers().values()

    def name_layers(self):
        """
        Returns an iterator over all layers in the network.

        Include name of the layer and layer itself.
        """

        value_set = set()
        layers = OrderedDict()
        for name, layer in self._layers.items():
            if layer is not None and layer not in value_set:
                value_set.add(layer)
                layers[name] = layer
        return layers

    def init_build(self, *inputs, **kwargs):
        """
        (1) This method must be called when the Layer has no input in_channels.
        (2) Automatic shape inference when the user does not enter in_channels.
        """

        self.forward(*inputs, **kwargs)

    def str_to_init(self, initializer):
        return str2init(initializer)

    def check_param(self, param, dim='2d'):
        return check_parameter(param, dim)

    def build_graph(self, *inputs, **kwargs):
        # Add nodes only when the composition is needed.
        for layer_name, layer in self._layers.items():
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
                    self.insert_child_to_layer(str(index), layer)
            elif isinstance(layers, OrderedDict):
                for name, layer in layers.items():
                    self.insert_child_to_layer(name, layer)
            else:
                raise TypeError('Layers must be list or orderedDict')
        else:
            for index, layer in enumerate(args):
                self.insert_child_to_layer(str(index), layer)
        self.layer_list = list(self._layers.values())

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(OrderedDict(list(self._layers.items())[index]))
        index = _valid_index(len(self), index)
        return list(self._layers.values())[index]

    def __setitem__(self, index, layer):
        if _valid_module(layer):
            index = _valid_index(len(self), index)
            key = list(self._layers.keys())[index]
            self._layers[key] = layer
            self.layer_list = list(self._layers.values())

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            key = list(self._layers.keys())[index]
            del self._layers[key]
        elif isinstance(index, slice):
            keys = list(self._layers.keys())[index]
            for key in keys:
                del self._layers[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        self.layer_list = list(self._layers.values())

    def __len__(self):
        return len(self._layers)

    def append(self, layer):
        if _valid_module(layer):
            self._layers[str(len(self))] = layer
        self.layer_list = list(self._layers.values())
        return self

    def build(self, inputs_shape):
        pass

    def forward(self, input_data):
        for layer in self.layer_list:
            input_data = layer(input_data)
        return input_data


class ModuleList(Module):
    """
    Holds submodules in a list.

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
        Initializing the ModuleList.
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
            return self.__class__(list(self._layers.values())[index])
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            return self._layers[str(index)]
        raise TypeError('Index {} is not int type or slice type'.format(index))

    def __setitem__(self, index, layer):
        if not isinstance(index, int) and _valid_module(layer):
            raise TypeError('Index {} is not int type'.format(index))
        index = _valid_index(len(self), index)
        self._layers[str(index)] = layer

    def __delitem__(self, index):
        if isinstance(index, int):
            index = _valid_index(len(self), index)
            del self._layers[str(index)]
        elif isinstance(index, slice):
            keys = list(self._layers.keys())[index]
            for key in keys:
                del self._layers[key]
        else:
            raise TypeError('Index {} is not int type or slice type'.format(index))
        temp_dict = OrderedDict()
        for idx, layer in enumerate(self._layers.values()):
            temp_dict[str(idx)] = layer
        self._layers = temp_dict

    def __len__(self):
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers.values())

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
            self._layers[str(length)] = self._layers[str(length - 1)]
            length -= 1
        self._layers[str(idx)] = layer

    def extend(self, layers):
        """
            Appends layers from a Python iterable to the end of the list.

        """

        if not isinstance(layers, list):
            raise TypeError('Modules {} should be list of sublayers'.format(layers))
        for layer in layers:
            if _valid_module(layer):
                self._layers[str(len(self))] = layer
        return self

    def append(self, layer):
        """
            Appends a given layer to the end of the list.

        """

        if _valid_module(layer):
            self._layers[str(len(self))] = layer

    def forward(self, *inputs):
        raise NotImplementedError


class ModuleDict(Module):
    """
    Holds submodules in a dictionary.

    ModuleDict can be used like a regular Python dictionary, support
    '__getitem__', '__setitem__', '__delitem__', '__len__', '__iter__' and '__contains__',
    but module it contains are properly registered, and will be visible by all Modules methods.

    Parameters
    ----------
        args : dict
            a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Methods
    ---------
    __init__()
        Initializing the ModuleDict.
    clear()
        Remove all items from the ModuleDict.
    pop()
        Remove key from the ModuleDict and return its module.
    keys()
        Return an iterable of the ModuleDict keys.
    items()
        Return an iterable of the ModuleDict key/value pairs.
    values()
        Return an iterable of the ModuleDict values.
    update()
        Update the ModuleDict with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

    Examples
    ---------
    >>> from tensorlayerx.nn import Module, ModuleDict, Linear
    >>> import tensorlayerx as tlx
    >>> class MyModule(Module):
    >>>     def __init__(self):
    >>>         super(MyModule, self).__init__()
    >>>         self.dict = ModuleDict({
    >>>                 'linear1':Linear(out_features=800, act=tlx.ReLU, in_features=784, name='linear1'),
    >>>                 'linear2':Linear(out_features=800, act=tlx.ReLU, in_features=800, name='linear2')
    >>>                 })
    >>>     def forward(self, x, linear):
    >>>         x = self.dict[linear](x)
    >>>         return x

    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):

        return self._layers[key]

    def __setitem__(self, key, module):
        if not isinstance(key, str):
            raise TypeError("module name should be a string, but got {}".format(type(key)))
        elif '.' in key:
            raise KeyError("module name can't contain \".\", got: {}".format(key))
        elif key == '':
            raise KeyError("module name can't be empty string \"\"")
        if _valid_module(module):
            self._layers[key] = module

    def __delitem__(self, key):

        del self._layers[key]

    def __len__(self):

        return len(self._layers)

    def __iter__(self):

        return iter(self._layers)

    def __contains__(self, key):

        return key in self._layers

    def clear(self):

        self._layers.clear()

    def pop(self, key):

        temp = self[key]
        del self[key]
        return temp

    def keys(self):

        return self._layers.keys()

    def items(self):

        return self._layers.items()

    def values(self):

        return self._layers.values()

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


class Parameter(Module):
    """This function creates a parameter. The parameter is a learnable variable, which can have gradient, and can be optimized.

    Parameters
    ----------
    data : Tensor
        parameter tensor
    requires_grad : bool
        if the parameter requires gradient. Default: True

    Returns
    -------
        Parameter

    Examples
    ----------
    >>> import tensorlayerx as tlx
    >>> para = tlx.nn.Parameter(data=tlx.ones((5,5)), requires_grad=True)

    """

    def __new__(self, data=None, name=None):
        instance = super().__new__(self)
        if name is None:
            prefix = 'parameter'

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
        if data is None:
            return instance
        else:
            return instance(data, name)

    def __call__(self, data=None, name=None, **kwargs):
        return tf.Variable(initial_value=data, name=name)


class ParameterList(Module):
    """Holds parameters in a list.

    ParameterList can be indexed like a regular Python list. Support
    '__getitem__', '__setitem__', '__delitem__', '__len__', '__iter__' and '__iadd__'.

    Parameters
    ----------
        Parameters : list
            List of Parameter.
    Methods
    ---------
    __init__()
        Initializing the ParameterList.
    extend(parameter)
        Appends parameters from a Python iterable to the end of the list.
    append(parameters)
        Appends a given parameter to the end of the list.

    Examples
    ---------
    >>> from tensorlayerx.nn import Module, ModuleList, Linear
    >>> import tensorlayerx as tlx
    >>> class MyModule(Module):
    >>>     def __init__(self):
    >>>         super(MyModule, self).__init__()
    >>>         self.params2 = ParameterList([Parameter(tlx.ones((10,5))), Parameter(tlx.ones((5,10)))])
    >>>     def forward(self, x):
    >>>         x = tlx.matmul(x, self.params2[0])
    >>>         x = tlx.matmul(x, self.params2[1])
    >>>         return x
    """

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
        # if not hasattr(self, key) and not isinstance(value, tf.Variable):
        #     warnings.warn("Setting attributes on ParameterList is not supported.")
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
        self.insert_param_to_layer(str(len(self)), parameter)
        return self

    def extend(self, parameters):
        if not isinstance(parameters, container_abcs.Iterable):
            raise TypeError(
                "ParameterList.extend should be called with an "
                "iterable, but got " + type(parameters).__name__
            )
        offset = len(self)
        for i, para in enumerate(parameters):
            self.insert_param_to_layer(str(offset + i), para)
        return self

    def __call__(self, input):
        raise RuntimeError('ParameterList should not be called.')


class ParameterDict(Module):
    """
    Holds parameters in a dictionary.

    ParameterDict can be used like a regular Python dictionary, support
    '__getitem__', '__setitem__', '__delitem__', '__len__', '__iter__' and '__contains__',


    Parameters
    ----------
        parameters : dict
            a mapping (dictionary) of (string: parameter)
            or an iterable of key-value pairs of type (string, parameter)

    Methods
    ---------
    __init__()
        Initializing the ParameterDict.
    clear()
        Remove all items from the ParameterDict.
    setdefault(key, default=None)
        If key is in the ParameterDict, return its parameter.
        If not, insert `key` with a parameter `default` and return `default`.
        `default` defaults to `None`.
    popitem()
        Remove and return the last inserted `(key, parameter)` pair from the ParameterDict
    pop(key)
        Remove key from the ParameterDict and return its parameter.
    get(key, default = None):
        Return the parameter associated with key if present. Otherwise return default if provided, None if not.
    fromkeys(keys, default = None)
        Return a new ParameterDict with the keys provided
    keys()
        Return an iterable of the ParameterDict keys.
    items()
        Return an iterable of the ParameterDict key/value pairs.
    values()
        Return an iterable of the ParameterDict values.
    update()
        Update the ParameterDict with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

    Examples
    ---------
    >>> from tensorlayerx.nn import Module, ParameterDict, Parameter
    >>> import tensorlayerx as tlx
    >>> class MyModule(Module):
    >>>     def __init__(self):
    >>>         super(MyModule, self).__init__()
    >>>         self.dict = ParameterDict({
    >>>                 'left': Parameter(tlx.ones((5, 10))),
    >>>                 'right': Parameter(tlx.zeros((5, 10)))
    >>>                 })
    >>>     def forward(self, x, choice):
    >>>         x = tlx.matmul(x, self.dict[choice])
    >>>         return x

    """

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, parameter):
        self.insert_param_to_layer(key, parameter)

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
