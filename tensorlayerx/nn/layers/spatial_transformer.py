#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from six.moves import xrange
import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    'transformer',
    'batch_transformer',
    'SpatialTransformer2dAffine',
]


def transformer(U, theta, out_size, name='SpatialTransformer2dAffine'):
    """Spatial Transformer Layer for `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__
    , see :class:`SpatialTransformer2dAffine` class.

    Parameters
    ----------
    U : list of float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the localisation network should be [num_batch, 6], value range should be [0, 1] (via tanh).
    out_size: tuple of int
        The size of the output of the network (height, width)
    name: str
        Optional function name

    Returns
    -------
    Tensor
        The transformed tensor.

    References
    ----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`__

    Notes
    -----
    To initialize the network to the identity transform init.

    >>> import tensorflow as tf
    >>> # ``theta`` to
    >>> identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    >>> identity = identity.flatten()
    >>> theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):
        rep = tlx.transpose(a=tlx.expand_dims(tlx.ones(shape=tlx.stack([
            n_repeats,
        ])), axis=1), perm=[1, 0])
        rep = tlx.cast(rep, 'int32')
        x = tlx.matmul(tlx.reshape(x, (-1, 1)), rep)
        return tlx.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, height, width, channels = tlx.get_tensor_shape(im)
        x = tlx.cast(x, 'float32')
        y = tlx.cast(y, 'float32')
        height_f = tlx.cast(height, 'float32')
        width_f = tlx.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tlx.zeros([], dtype='int32')
        max_y = tlx.cast(height - 1, 'int32')
        max_x = tlx.cast(width - 1, 'int32')

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = tlx.cast(tlx.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tlx.cast(tlx.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tlx.clip_by_value(x0, zero, max_x)
        x1 = tlx.clip_by_value(x1, zero, max_x)
        y0 = tlx.clip_by_value(y0, zero, max_y)
        y1 = tlx.clip_by_value(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(tlx.range(num_batch) * dim1, out_height * out_width)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tlx.reshape(im, tlx.stack([-1, channels]))
        im_flat = tlx.cast(im_flat, 'float32')
        Ia = tlx.gather(im_flat, idx_a)
        Ib = tlx.gather(im_flat, idx_b)
        Ic = tlx.gather(im_flat, idx_c)
        Id = tlx.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tlx.cast(x0, 'float32')
        x1_f = tlx.cast(x1, 'float32')
        y0_f = tlx.cast(y0, 'float32')
        y1_f = tlx.cast(y1, 'float32')
        wa = tlx.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = tlx.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = tlx.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = tlx.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = tlx.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tlx.matmul(
            tlx.ones(shape=tlx.stack([height, 1])),
            tlx.transpose(a=tlx.expand_dims(tlx.linspace(-1.0, 1.0, width), 1), perm=[1, 0])
        )
        y_t = tlx.matmul(tlx.expand_dims(tlx.linspace(-1.0, 1.0, height), 1), tlx.ones(shape=tlx.stack([1, width])))

        x_t_flat = tlx.reshape(x_t, (1, -1))
        y_t_flat = tlx.reshape(y_t, (1, -1))

        ones = tlx.ones(shape=tlx.get_tensor_shape(x_t_flat))
        grid = tlx.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
        return grid

    def _transform(theta, input_dim, out_size):
        num_batch, _, _, num_channels = tlx.get_tensor_shape(input_dim)
        theta = tlx.reshape(theta, (-1, 2, 3))
        theta = tlx.cast(theta, 'float32')

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width)
        grid = tlx.expand_dims(grid, 0)
        grid = tlx.reshape(grid, [-1])
        grid = tlx.tile(grid, tlx.stack([num_batch]))
        grid = tlx.reshape(grid, tlx.stack([num_batch, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tlx.matmul(theta, grid)
        x_s = tlx.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tlx.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tlx.reshape(x_s, [-1])
        y_s_flat = tlx.reshape(y_s, [-1])

        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

        output = tlx.reshape(input_transformed, tlx.stack([num_batch, out_height, out_width, num_channels]))
        return output

    output = _transform(theta, U, out_size)
    return output


def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer2dAffine'):
    """Batch Spatial Transformer function for `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__.

    Parameters
    ----------
    U : list of float
        tensor of inputs [batch, height, width, num_channels]
    thetas : list of float
        a set of transformations for each input [batch, num_transforms, 6]
    out_size : list of int
        the size of the output [out_height, out_width]
    name : str
        optional function name

    Returns
    ------
    float
        Tensor of size [batch * num_transforms, out_height, out_width, num_channels]

    """
    # with tf.compat.v1.variable_scope(name):
    num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
    indices = [[i] * num_transforms for i in xrange(num_batch)]
    input_repeated = tlx.gather(U, tlx.reshape(indices, [-1]))
    return transformer(input_repeated, thetas, out_size)


class SpatialTransformer2dAffine(Module):
    """The :class:`SpatialTransformer2dAffine` class is a 2D `Spatial Transformer Layer <https://arxiv.org/abs/1506.02025>`__ for
    `2D Affine Transformation <https://en.wikipedia.org/wiki/Affine_transformation>`__.

    Parameters
    -----------
    out_size : tuple of int or None
        - The size of the output of the network (height, width), the feature maps will be resized by this.
    in_channels : int
        The number of in channels.
    data_format : str
        "channel_last" (NHWC, default) or "channels_first" (NCHW).
    name : str
        - A unique layer name.

    References
    -----------
    - `Spatial Transformer Networks <https://arxiv.org/abs/1506.02025>`__
    - `TensorFlow/Models <https://github.com/tensorflow/models/tree/master/transformer>`__

    """

    def __init__(
        self,
        out_size=(40, 40),
        in_channels=None,
        data_format='channel_last',
        name=None,
    ):
        super(SpatialTransformer2dAffine, self).__init__(name)

        self.in_channels = in_channels
        self.out_size = out_size
        self.data_format = data_format
        if self.in_channels is not None:
            self.build(self.in_channels)
            self._built = True

        logging.info("SpatialTransformer2dAffine %s" % self.name)

    def __repr__(self):
        s = '{classname}(out_size={out_size}, '
        if self.in_channels is not None:
            s += 'in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None and len(inputs_shape) != 2:
            raise AssertionError("The dimension of theta layer input must be rank 2, please reshape or flatten it")
        if self.in_channels:
            shape = [self.in_channels, 6]
        else:
            # self.in_channels = inputs_shape[1]    # BUG
            # shape = [inputs_shape[1], 6]
            self.in_channels = inputs_shape[0][-1]  # zsdonghao
            shape = [self.in_channels, 6]
        self.W = self._get_weights("weights", shape=tuple(shape), init=tlx.nn.initializers.Zeros())
        identity = np.reshape(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), newshape=(6, ))
        self.b = self._get_weights("biases", shape=(6, ), init=tlx.nn.initializers.Constant(identity))

    def forward(self, inputs):
        """
        :param inputs: a tuple (theta_input, U).
                    - theta_input is of size [batch, in_channels]. We will use a :class:`Dense` to
                    make the theta size to [batch, 6], value range to [0, 1] (via tanh).
                    - U is the previous layer, which the affine transformation is applied to.
        :return: tensor of size [batch, out_size[0], out_size[1], n_channels] after affine transformation,
                    n_channels is identical to that of U.
        """
        theta_input, U = inputs
        theta = tlx.tanh(tlx.matmul(theta_input, self.W) + self.b)
        outputs = transformer(U, theta, out_size=self.out_size)
        # automatically set batch_size and channels
        # e.g. [?, 40, 40, ?] --> [64, 40, 40, 1] or [64, 20, 20, 4]
        batch_size = theta_input.shape[0]
        n_channels = U.shape[-1]
        if self.data_format == 'channel_last':
            outputs = tlx.reshape(outputs, shape=[batch_size, self.out_size[0], self.out_size[1], n_channels])
        else:
            raise Exception("unimplement data_format {}".format(self.data_format))

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, outputs)
            self._nodes_fixed = True
        return outputs
