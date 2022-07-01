#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import tensorlayerx as tlx
from tensorlayerx.files import utils
from tensorlayerx import logging
import numpy as np
from queue import Queue
from tensorlayerx.nn.initializers import *

if tlx.BACKEND == 'mindspore':
    from mindspore.ops.operations import Assign
    from mindspore.nn import Cell
    from mindspore import Tensor
    import mindspore as ms

_act_dict = {
    "relu": tlx.ops.ReLU,
    "relu6": tlx.ops.ReLU6,
    "leaky_relu": tlx.ops.LeakyReLU,
    "lrelu": tlx.ops.LeakyReLU,
    "softplus": tlx.ops.Softplus,
    "tanh": tlx.ops.Tanh,
    "sigmoid": tlx.ops.Sigmoid,
    "softmax": tlx.ops.Softmax
}

_initializers_dict = {
    "ones": ones(),
    "zeros": zeros(),
    "constant": constant(value=0.0),
    "random_uniform": random_uniform(minval=-1.0, maxval=1.0),
    "random_normal": random_normal(mean=0.0, stddev=0.05),
    "truncated_normal": truncated_normal(stddev=0.02),
    "he_normal": he_normal(),
    "xavier_uniform": XavierUniform(),
    "xavier_normal": XavierNormal()
}


def check_parameter(parameter, dim='2d'):
    if dim == '2d':
        if isinstance(parameter, int):
            out = (parameter, parameter)
        else:
            out = parameter
    elif dim == '3d':
        if isinstance(parameter, int):
            out = (parameter, parameter, parameter)
        else:
            out = parameter
    else:
        raise ("dim must be 2d or 3d.")
    return out


def str2init(initializer):
    if isinstance(initializer, str):
        if initializer not in _initializers_dict.keys():
            raise Exception(
                "Unsupported string initialization: {}".format(initializer),
                "String initialization supports these methods: {}".format(_initializers_dict.keys())
            )
        return _initializers_dict[initializer]
    else:
        return initializer


def str2act(act):
    if len(act) > 5 and act[0:5] == "lrelu":
        try:
            alpha = float(act[5:])
            return tlx.ops.LeakyReLU(negative_slope=alpha)
        except Exception as e:
            raise Exception("{} can not be parsed as a float".format(act[5:]))

    if len(act) > 10 and act[0:10] == "leaky_relu":
        try:
            alpha = float(act[10:])
            return tlx.ops.LeakyReLU(negative_slope=alpha)
        except Exception as e:
            raise Exception("{} can not be parsed as a float".format(act[10:]))

    if act not in _act_dict.keys():
        raise Exception("Unsupported act: {}".format(act))
    return _act_dict[act]


def processing_act(act):
    # Processing strings as input, activation functions without parameters。
    if isinstance(act, str):
        str_act = str2act(act)
    if act:
        # Processing strings as input, activation functions with parameters。
        if isinstance(act, str) and (len(act) > 5 and act[0:5] == "lrelu" or
                                     len(act) > 10 and act[0:10] == "leaky_relu"):
            out_act = str_act
        elif isinstance(act, str):
            out_act = str_act()
        else:
            # Processing classes or functions as input, activation functions without parameters
            try:
                out_act = act()
            # Processing class or function as input, activation function with parameters
            except:
                out_act = act
    else:
        # Processing act is None
        out_act = act
    return out_act


def _save_weights(net, file_path, format=None):
    """Input file_path, save model weights into a file of given format.
                Use net.load_weights() to restore.

    Parameters
    ----------
    file_path : str
        Filename to which the model weights will be saved.
    format : str or None
        Saved file format.
        Value should be None, 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
        1) If this is set to None, then the postfix of file_path will be used to decide saved format.
        If the postfix is not in ['h5', 'hdf5', 'npz', 'ckpt'], then file will be saved in hdf5 format by default.
        2) 'hdf5' will save model weights name in a list and each layer has its weights stored in a group of
        the hdf5 file.
        3) 'npz' will save model weights sequentially into a npz file.
        4) 'npz_dict' will save model weights along with its name as a dict into a npz file.
        5) 'ckpt' will save model weights into a tensorflow ckpt file.

        Default None.

    Examples
    --------
    1) Save model weights in hdf5 format by default.
    >>> net = vgg16()
    >>> optimizer = tlx.optimizers.Adam(learning_rate=0.001)
    >>> metrics = tlx.metrics.Accuracy()
    >>> model = tlx.model.Model(network=net, loss_fn=tlx.losses.cross_entropy, optimizer=optimizer, metrics=metrics)
    >>> model.save_weights('./model.h5')
    ...
    >>> model.load_weights('./model.h5')

    2) Save model weights in npz/npz_dict format
    >>> model.save_weights('./model.npz')
    >>> model.save_weights('./model.npz', format='npz_dict')

    """

    if tlx.BACKEND != 'torch' and net.all_weights is None or len(net.all_weights) == 0:
        logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
        return

    if format is None:
        postfix = file_path.split('.')[-1]
        if postfix in ['h5', 'hdf5', 'npz', 'ckpt']:
            format = postfix
        else:
            format = 'hdf5'

    if format == 'hdf5' or format == 'h5':
        raise NotImplementedError("hdf5 load/save is not supported now.")
        # utils.save_weights_to_hdf5(file_path, net)
    elif format == 'npz':
        utils.save_npz(net.all_weights, file_path)
    elif format == 'npz_dict':
        if tlx.BACKEND == 'torch':
            utils.save_npz_dict(net.named_parameters(), file_path)
        else:
            utils.save_npz_dict(net.all_weights, file_path)
    elif format == 'ckpt':
        # TODO: enable this when tf save ckpt is enabled
        raise NotImplementedError("ckpt load/save is not supported now.")
    else:
        raise ValueError(
            "Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
            "Other format is not supported now."
        )


def _load_weights(net, file_path, format=None, in_order=True, skip=False):
    """Load model weights from a given file, which should be previously saved by net.save_weights().

    Parameters
    ----------
    file_path : str
        Filename from which the model weights will be loaded.
    format : str or None
        If not specified (None), the postfix of the file_path will be used to decide its format. If specified,
        value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
        In addition, it should be the same format when you saved the file using net.save_weights().
        Default is None.
    in_order : bool
        Allow loading weights into model in a sequential way or by name. Only useful when 'format' is 'hdf5'.
        If 'in_order' is True, weights from the file will be loaded into model in a sequential way.
        If 'in_order' is False, weights from the file will be loaded into model by matching the name
        with the weights of the model, particularly useful when trying to restore model in eager(graph) mode from
        a weights file which is saved in graph(eager) mode.
        Default is True.
    skip : bool
        Allow skipping weights whose name is mismatched between the file and model. Only useful when 'format' is
        'hdf5' or 'npz_dict'. If 'skip' is True, 'in_order' argument will be ignored and those loaded weights
        whose name is not found in model weights (net.all_weights) will be skipped. If 'skip' is False, error will
        occur when mismatch is found.
        Default is False.

    Examples
    --------
    1) load model from a hdf5 file.
    >>> net = vgg16()
    >>> optimizer = tlx.optimizers.Adam(learning_rate=0.001)
    >>> metrics = tlx.metrics.Accuracy()
    >>> model = tlx.model.Model(network=net, loss_fn=tlx.losses.cross_entropy, optimizer=optimizer, metrics=metrics)
    >>> model.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
    >>> model.load_weights('./model_eager.h5') # load sequentially

    2) load model from a npz file
    >>> model.load_weights('./model.npz')

    3) load model from a npz file, which is saved as npz_dict previously
    >>> model.load_weights('./model.npz', format='npz_dict')

    Notes
    -------
    1) 'in_order' is only useful when 'format' is 'hdf5'. If you are trying to load a weights file which is
       saved in a different mode, it is recommended to set 'in_order' be True.
    2) 'skip' is useful when 'format' is 'hdf5' or 'npz_dict'. If 'skip' is True,
       'in_order' argument will be ignored.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("file {} doesn't exist.".format(file_path))

    if format is None:
        format = file_path.split('.')[-1]

    if format == 'hdf5' or format == 'h5':
        raise NotImplementedError("hdf5 load/save is not supported now.")
        # if skip ==True or in_order == False:
        #     # load by weights name
        #     utils.load_hdf5_to_weights(file_path, net, skip)
        # else:
        #     # load in order
        #     utils.load_hdf5_to_weights_in_order(file_path, net)
    elif format == 'npz':
        utils.load_and_assign_npz(file_path, net)
    elif format == 'npz_dict':
        utils.load_and_assign_npz_dict(file_path, net, skip)
    elif format == 'ckpt':
        # TODO: enable this when tf save ckpt is enabled
        raise NotImplementedError("ckpt load/save is not supported now.")
    else:
        raise ValueError(
            "File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
            "Other format is not supported now."
        )


def _save_standard_weights_dict(net, file_path):
    # Eliminate parameter naming differences between frameworks.
    if tlx.BACKEND == 'torch':
        save_standard_npz_dict(net.named_parameters(), file_path)
    else:
        save_standard_npz_dict(net.all_weights, file_path)


def encode_list_name(list_name):
    # TensorFlow weights format: conv1.weight:0, conv1.bias:0
    # Paddle weights format: conv1.weight, conv1.bias
    # PyTorch weights format: conv1.W, conv1.W
    # MindSpore weights format: conv1.weights, conv1.bias
    # standard weights format: conv1.weights, conv1.bias

    for i in range(len(list_name)):
        if tlx.BACKEND == 'tensorflow':
            list_name[i] = list_name[i][:-2]
        if tlx.BACKEND == 'torch':
            if list_name[i][-1] == 'W' and 'conv' not in list_name[i]:
                list_name[i] = list_name[i][:-2] + str('/weights')
            elif list_name[i][-1] == 'W' and 'conv' in list_name[i]:
                list_name[i] = list_name[i][:-2] + str('/filters')
            elif list_name[i][-1] == 'b':
                list_name[i] = list_name[i][:-2] + str('/biases')
            elif list_name[i].split('.')[-1] in ['beta', 'gamma', 'moving_mean', 'moving_var']:
                pass
            else:
                raise NotImplementedError('This weights cannot be converted.')
    return list_name


def decode_key_name(key_name):
    if tlx.BACKEND == 'tensorflow':
        key_name = key_name + str(':0')
    if tlx.BACKEND == 'torch':
        if key_name.split('/')[-1] in ['weights', 'filters']:
            key_name = key_name[:-8] + str('.W')
        elif key_name.split('/')[-1] == 'biases':
            key_name = key_name[:-7] + str('.b')
        else:
            raise NotImplementedError('This weights cannot be converted.')
    return key_name


def save_standard_npz_dict(save_list=None, name='model.npz'):
    """Input parameters and the file name, save parameters as a dictionary into standard npz_dict file.

    Use ``tlx.files.load_and_assign_npz_dict()`` to restore.

    Parameters
    ----------
    save_list : list of parameters
        A list of parameters (tensor) to be saved.
    name : str
        The name of the `.npz` file.

    """
    if save_list is None:
        save_list = []
    if tlx.BACKEND != 'torch':
        save_list_names = [tensor.name for tensor in save_list]

    if tlx.BACKEND == 'tensorflow':
        save_list_var = utils.tf_variables_to_numpy(save_list)
    elif tlx.BACKEND == 'mindspore':
        save_list_var = utils.ms_variables_to_numpy(save_list)
    elif tlx.BACKEND == 'paddle':
        save_list_var = utils.pd_variables_to_numpy(save_list)
    elif tlx.BACKEND == 'torch':
        save_list_names = []
        save_list_var = []
        for named, values in save_list:
            save_list_names.append(named)
            save_list_var.append(values.cpu().detach().numpy())
    else:
        raise NotImplementedError('Not implemented')

    save_list_names = encode_list_name(save_list_names)

    save_var_dict = {save_list_names[idx]: val for idx, val in enumerate(save_list_var)}
    np.savez(name, **save_var_dict)
    save_list_var = None
    save_var_dict = None
    del save_list_var
    del save_var_dict
    logging.info("[*] Model saved in npz_dict %s" % name)


def _load_standard_weights_dict(net, file_path, skip=False, reshape=False, format='npz_dict'):
    if format == 'npz_dict':
        load_and_assign_standard_npz_dict(net, file_path, skip, reshape)
    elif format == 'npz':
        load_and_assign_standard_npz(file_path, net, reshape)


def load_and_assign_standard_npz_dict(net, file_path, skip=False, reshape=False):
    if not os.path.exists(file_path):
        logging.error("file {} doesn't exist.".format(file_path))
        return False

    weights = np.load(file_path, allow_pickle=True)
    if len(weights.keys()) != len(set(weights.keys())):
        raise Exception("Duplication in model npz_dict %s" % file_path)

    if tlx.BACKEND == 'torch':
        net_weights_name = [n for n, v in net.named_parameters()]
        torch_weights_dict = {n: v for n, v in net.named_parameters()}
    else:
        net_weights_name = [w.name for w in net.all_weights]

    for key in weights.keys():
        de_key = decode_key_name(key)
        if de_key not in net_weights_name:
            if skip:
                logging.warning("Weights named '%s' not found in network. Skip it." % key)
            else:
                raise RuntimeError(
                    "Weights named '%s' not found in network. Hint: set argument skip=Ture "
                    "if you want to skip redundant or mismatch weights." % key
                )
        else:
            if tlx.BACKEND == 'tensorflow':
                reshape_weights = weight_reshape(weights[key], reshape)
                check_reshape(reshape_weights, net.all_weights[net_weights_name.index(de_key)])
                utils.assign_tf_variable(net.all_weights[net_weights_name.index(de_key)], reshape_weights)
            elif tlx.BACKEND == 'mindspore':
                reshape_weights = weight_reshape(weights[key], reshape)
                import mindspore as ms
                assign_param = ms.Tensor(reshape_weights, dtype=ms.float32)
                check_reshape(assign_param, net.all_weights[net_weights_name.index(de_key)])
                utils.assign_ms_variable(net.all_weights[net_weights_name.index(de_key)], assign_param)
            elif tlx.BACKEND == 'paddle':
                reshape_weights = weight_reshape(weights[key], reshape)
                check_reshape(reshape_weights, net.all_weights[net_weights_name.index(de_key)])
                utils.assign_pd_variable(net.all_weights[net_weights_name.index(de_key)], reshape_weights)
            elif tlx.BACKEND == 'torch':
                reshape_weights = weight_reshape(weights[key], reshape)
                check_reshape(reshape_weights, net.all_weights[net_weights_name.index(de_key)])
                utils.assign_th_variable(torch_weights_dict[de_key], reshape_weights)
            else:
                raise NotImplementedError('Not implemented')

    logging.info("[*] Model restored from npz_dict %s" % file_path)


def load_and_assign_standard_npz(file_path=None, network=None, reshape=False):
    if network is None:
        raise ValueError("network is None.")

    if not os.path.exists(file_path):
        logging.error("file {} doesn't exist.".format(file_path))
        return False
    else:
        weights = utils.load_npz(name=file_path)
        ops = []
        if tlx.BACKEND == 'tensorflow':
            for idx, param in enumerate(weights):
                param = weight_reshape(param, reshape)
                check_reshape(param, network.all_weights[idx])
                ops.append(network.all_weights[idx].assign(param))

        elif tlx.BACKEND == 'mindspore':

            class Assign_net(Cell):

                def __init__(self, y):
                    super(Assign_net, self).__init__()
                    self.y = y

                def construct(self, x):
                    Assign()(self.y, x)

            for idx, param in enumerate(weights):
                assign_param = Tensor(param, dtype=ms.float32)
                assign_param = weight_reshape(assign_param, reshape)
                check_reshape(assign_param, network.all_weights[idx])
                Assign()(network.all_weights[idx], assign_param)

        elif tlx.BACKEND == 'paddle':
            for idx, param in enumerate(weights):
                param = weight_reshape(param, reshape)
                check_reshape(param, network.all_weights[idx])
                utils.assign_pd_variable(network.all_weights[idx], param)

        elif tlx.BACKEND == 'torch':
            for idx, param in enumerate(weights):
                param = weight_reshape(param, reshape)
                check_reshape(param, network.all_weights[idx])
                utils.assign_th_variable(network.all_weights[idx], param)
        else:
            raise NotImplementedError("This backend is not supported")
        return ops

    logging.info("[*] Load {} SUCCESS!".format(file_path))


def check_reshape(weight, shape_weights):
    if len(weight.shape) >= 4 and weight.shape[::-1] == tuple(shape_weights.shape):
        if tlx.BACKEND == 'tensorflow':

            raise Warning(
                'Set reshape to True only when importing weights from MindSpore/PyTorch/PaddlePaddle to TensorFlow.'
            )
        if tlx.BACKEND == 'torch':
            raise Warning('Set reshape to True only when importing weights from TensorFlow to PyTorch.')
        if tlx.BACKEND == 'paddle':
            raise Warning('Set reshape to True only when importing weights from TensorFlow to PaddlePaddle.')
        if tlx.BACKEND == 'mindspore':
            raise Warning('Set reshape to True only when importing weights from TensorFlow to MindSpore.')


def weight_reshape(weight, reshape=False):
    # TODO In this case only 2D convolution is considered. 3D convolution tests need to be supplemented.
    if reshape:
        if len(weight.shape) == 4:
            weight = np.moveaxis(weight, (2, 3), (1, 0))
        if len(weight.shape) == 5:
            weight = np.moveaxis(weight, (3, 4), (1, 0))
    return weight

def tolist(tensors):
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        ntensors = list()
        for t in tensors:
            ntensors += tolist(t)
        return ntensors
    else:
        return [tensors]

def construct_graph(inputs, outputs):
    """construct computation graph for model using ModuleNode object"""
    all_layers = []
    node_by_depth = []

    input_tensors_list = inputs if isinstance(inputs, list) else inputs

    queue_node = Queue()
    # BFS to visit all nodes that should be involved in the computation graph
    output_tensors_list = outputs if isinstance(outputs, list) else [outputs]
    output_nodes = [tensor._info[0] for tensor in output_tensors_list]

    visited_node_names = set()
    for out_node in output_nodes:
        if out_node.visited:
            continue
        queue_node.put(out_node)

        while not queue_node.empty():
            cur_node = queue_node.get()
            in_nodes = cur_node.in_nodes

            for node in in_nodes:
                node.out_nodes.append(cur_node)
                if not node.visited:
                    queue_node.put(node)
                    node.visited = True
                    if node.node_name not in visited_node_names:
                        visited_node_names.add(node.node_name)
                    # else have multiple layers with the same name
                    else:
                        raise ValueError(
                            'Layer name \'%s\' has already been used by another layer. Please change the layer name.'
                            % node.layer.name
                        )

    # construct the computation graph in top-sort order
    cur_depth = [tensor._info[0] for tensor in input_tensors_list]
    next_depth = []
    indegrees = {}

    visited_layer_names = []
    while not len(cur_depth) == 0:
        node_by_depth.append(cur_depth)
        for node in cur_depth:
            if node.layer.name not in visited_layer_names:
                all_layers.append(node.layer)
                visited_layer_names.append(node.layer.name)
            for out_node in node.out_nodes:
                if out_node.node_name not in indegrees.keys():
                    indegrees[out_node.node_name] = len(out_node.in_nodes)
                indegrees[out_node.node_name] -= 1
                if indegrees[out_node.node_name] == 0 or \
                        isinstance(out_node.layer, (tlx.nn.RNN, tlx.nn.LSTM, tlx.nn.GRU)):
                    next_depth.append(out_node)
        cur_depth = next_depth
        next_depth = []
    return node_by_depth, all_layers


def select_attrs(obj):
    attrs_dict = obj.__dict__
    attrs = {}
    _select_key = ['kernel_size', 'stride', 'act', 'padding', 'data_format', 'concat_dim', 'dilation', 'bias']
    for k in _select_key:
        if k in attrs_dict:
            if k == 'act':
                attrs[k] = attrs_dict[k].__class__.__name__
            else:
                attrs[k] = attrs_dict[k]
    return attrs


class ModuleNode(object):
    """
    The class :class:`ModuleNode` class represents a conceptional node for a layer.

    ModuleNode is used for building topology and it is actually a light weighted
    wrapper over Layer.

    Parameters
    ----------
    layer : tl.layers.Layer
        A tl layer that wants to create a node.
    node_index : int
        Index of this node in layer._nodes.
    in_nodes ：a list of ModuleNode
        Father nodes to this node.
    in_tensors : a list of tensors
        Input tensors to this node.
    out_tensors : a list of tensors
        Output tensors to this node.
    in_tensor_idxes : a list of int
        Indexes of each input tensor in its corresponding node's out_tensors.

    Methods
    ---------
    __init__()
        Initializing the ModuleNode.
    __call__()
        (1) Forwarding through the layer. (2) Update its input/output tensors.
    """

    def __init__(self, layer, node_index, in_nodes, in_tensors, out_tensors, in_tensor_idxes, attr):
        self.layer = layer
        self.node_index = node_index
        self.in_nodes = in_nodes
        self.out_nodes = []
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors
        self.node_name = layer.name + "_node_{}".format(node_index)

        self.in_tensors_idxes = in_tensor_idxes
        self.attr = attr
        self.visited = False

    def __call__(self, inputs, **kwargs):
        """(1) Forwarding through the layer. (2) Update its input/output tensors."""
        outputs = self.layer(inputs, **kwargs)
        self.in_tensors = tolist(inputs)
        self.out_tensors = tolist(outputs)
        return self.out_tensors