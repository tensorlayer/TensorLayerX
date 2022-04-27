#! /usr/bin/python
# -*- coding: utf-8 -*-

from .common import str2act, str2init, tolist, construct_graph, ModuleNode
from .common import _save_weights, _load_weights, _save_standard_weights_dict, _load_standard_weights_dict
from collections import OrderedDict, abc as container_abcs
from collections import OrderedDict
import time
import tensorlayerx as tlx
import tensorflow as tf
from tensorlayerx.nn.layers.utils import (get_variable_with_initializer, random_normal)

__all__ = ['Module', 'Sequential', 'ModuleList', 'ModuleDict']

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
        self._all_weights = None
        self._trainable_weights = None
        self._nontrainable_weights = None

        # layer forward  state
        self._forward_state = False

        # Layer training state
        self.is_train = True

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

        elif isinstance(value, Module):
            if layers is None:
                raise AttributeError("Can not assign layers before Module.__init__() call.")
            if name in self.__dict__:
                del self.__dict__[name]
            if params and name in params:
                raise TypeError("Expected type is Parameter, but got Module.")
            # TODO Automatic shape inference when the user does not enter inchannels.
            # if value._built is False:
            #     raise AttributeError(
            #         "The registered layer `{}` should be built in advance. "
            #         "Do you forget to pass the keyword argument 'in_channels'? ".format(value.name)
            #     )
            layers[name] = value
        else:
            object.__setattr__(self, name, value)

    def __call__(self, inputs, *args, **kwargs):

        output = self.forward(inputs, *args, **kwargs)

        return output

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

    def build_graph(self, *inputs, **kwargs):
        # Add nodes only when the composition is needed.
        layers = self.layers_and_names(name_prefix='')
        for layer_name, layer in layers:
            if isinstance(layer, Module):
                layer._build_graph = True

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

    def __init__(self, modules = None):
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

    def __init__(self, modules = None):
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
