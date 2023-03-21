#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module

__all__ = [
    # 'Fold', #TODO
    'Unfold',
]

class Fold(Module):
    r"""Combines an array of sliding local blocks into a large containing
    tensor.

    Consider a batched :attr:`input` tensor containing sliding local blocks,
    e.g., patches of images, of shape :math:`(N, C \times  \prod(\text{kernel\_size}), L)`,
    where :math:`N` is batch dimension, :math:`C \times \prod(\text{kernel\_size})`
    is the number of values within a block (a block has :math:`\prod(\text{kernel\_size})`
    spatial locations each containing a :math:`C`-channeled vector), and
    :math:`L` is the total number of blocks. (This is exactly the
    same specification as the output shape of :class:`~torch.nn.Unfold`.) This
    operation combines these local blocks into the large :attr:`output` tensor
    of shape :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
    by summing the overlapping values. Similar to :class:`~torch.nn.Unfold`, the
    arguments must satisfy

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`d` is over all spatial dimensions.

    * :attr:`output_size` describes the spatial shape of the large containing
      tensor of the sliding local blocks. It is useful to resolve the ambiguity
      when multiple input shapes map to same number of sliding blocks, e.g.,
      with ``stride > 0``.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Parameters
    ----------
    output_size : int or tuple
        the shape of the spatial dimensions of the
                                    output (i.e., ``output.sizes()[2:]``)
    kernel_size : int or tuple
        the size of the sliding blocks
    stride : int or tuple
        the stride of the sliding blocks in the input spatial dimensions. Default: 1
    padding : int or tuple, optional
        implicit zero padding to be added on both sides of input. Default: 0
    dilation : int or tuple, optional
        a parameter that controls the stride of elements within the neighborhood. Default: 1

    .. warning::
        Currently, only 4-D output tensors (batched image-like tensors) are
        supported.

    Shape:
        - Input: :math:`(N, C \times \prod(\text{kernel\_size}), L)`
        - Output: :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)` as described above

    Examples
    ----------
    >>> import numpy as np
    >>> import tensorlayerx as tlx
    >>> fold = tlx.nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
    >>> input = tlx.convert_to_tensor( np.random.random(1, 3 * 2 * 2, 12))
    >>> output = fold(input)
    >>> output.shape

    """
    pass

class Unfold(Module):
    r"""Extracts sliding local blocks from a batched input tensor.

    Consider a batched :attr:`input` tensor of shape :math:`(N, C, *)`,
    where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
    and :math:`*` represent arbitrary spatial dimensions. This operation flattens
    each sliding :attr:`kernel_size`-sized block within the spatial dimensions
    of :attr:`input` into a column (i.e., last dimension) of a 3-D :attr:`output`
    tensor of shape :math:`(N, C \times \prod(\text{kernel\_size}), L)`, where
    :math:`C \times \prod(\text{kernel\_size})` is the total number of values
    within each block (a block has :math:`\prod(\text{kernel\_size})` spatial
    locations each containing a :math:`C`-channeled vector), and :math:`L` is
    the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`\text{spatial\_size}` is formed by the spatial dimensions
    of :attr:`input` (:math:`*` above), and :math:`d` is over all spatial
    dimensions.

    Therefore, indexing :attr:`output` at the last dimension (column dimension)
    gives all values within a certain block.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Parameters
    ----------
    kernel_size (int or tuple):
        the size of the sliding blocks
    stride : int or tuple, optional
        the stride of the sliding blocks in the input spatial dimensions. Default: 1
    padding : int or tuple, optional)
        implicit zero padding to be added on both sides of input. Default: 0
    dilation : int or tuple, optional
        a parameter that controls the stride of elements within the neighborhood. Default: 1

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C \times \prod(\text{kernel\_size}), L)` as described above

    Examples
    ----------
    >>> import numpy as np
    >>> import tensorlayerx as tlx
    >>> unfold = nn.Unfold(kernel_size=(2, 3))
    >>> input = tlx.convert_to_tensor(np.random.random(2, 5, 3, 4))
    >>> output = unfold(input)
    >>> # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
    >>> # 4 blocks (2x3 kernels) in total in the 3x4 input
    >>> output.size()

    """


    def __init__(
        self,
        kernel_size ,
        dilation = 1,
        padding = 0,
        stride = 1,
        name = 'unfold',
    ) -> None:
        super(Unfold, self).__init__()
        msg = "{} must be int or 2-tuple for 4D input"
        self.kernel_size = self.assert_int_or_pair(kernel_size, "kernel_size", msg)
        self.dilation = self.assert_int_or_pair(dilation, "dilation", msg)
        self.padding = self.assert_int_or_pair(padding, "padding", msg)
        self.stride = self.assert_int_or_pair(stride, "stride", msg)
        self.name = name
        self.build()
        self._built = True

    def assert_int_or_pair(self, arg, arg_name, message):
        assert isinstance(arg, int) or len(arg) == 2, message.format(arg_name)
        if isinstance(arg, int):
            return [arg,] * 2

    def __repr__(self):
        s = '{classname}('
        s += 'kernel_size={kernel_size},'
        s += 'dilation={dilation},'
        s += 'padding={padding},'
        s += 'stride={stride},'
        s += 'name={name}'
        s += ")"
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    def forward(self, input):
        if len(input.shape) != 4:
            raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(len(input.shape)))
        output = tlx.ops.unfold(input, self.kernel_size, self.dilation,
                        self.padding, self.stride)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(input, output)
            self._nodes_fixed = True
        return output
