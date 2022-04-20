##! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.nn.core import Module
from tensorlayerx.nn.core import ModuleList
import numpy as np

__all__ = [
    'MultiheadAttention',
    'Transformer',
    'TransformerEncoder',
    'TransformerDecoder',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
]


class MultiheadAttention(Module):
    """
    Allows the model to jointly attend to information from different representation subspaces.

    Parameters
    ----------
    embed_dim: int
        total dimension of the model.
    num_heads : int
        The number of heads in multi-head attention.
    dropout : float
        a Dropout layer on attn_output_weights. Default: 0.0.
    kdim : int
        total number of features in key. Default: None.
    vdim : int
        total number of features in value. Default: None.
    bias : bool
        add bias as module parameter. Default: True.
    batch_first: bool
        If ``True``, then the input and output tensors are provided as `[batch, seq, feature]`. Default: ``False`` `[seq, batch, feature]`.
    need_weights: bool
        Indicate whether to return the attention weights. Default ``False``.
    name: None or str
        A unique layer name.

    Examples
    ---------
    With TensorLayerX

    >>> q = tlx.nn.Input(shape=(4,2,128))
    >>> attn_mask = tlx.convert_to_tensor(np.zeros((4,4)),dtype='bool')
    >>> layer = MultiheadAttention(embed_dim=128, num_heads=4)
    >>> output = layer(q, attn_mask=attn_mask)


    References
    ----------
    - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`__

    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        bias=True,
        batch_first=False,
        need_weights=True,
        name=None,
    ):
        super(MultiheadAttention, self).__init__(name)

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        self.batch_first = batch_first
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.build(None)
        logging.info(
            "MultiheadAttention %s: embed_dim: %d num_heads: %d kdim: %d vdim: %d dropout: %f" %
            (self.name, embed_dim, num_heads, self.kdim, self.vdim, dropout)
        )

    def __repr__(self):
        s = (
            '{classname}(embed_dim={embed_dim}, num_heads={num_heads}, dropout={dropout}'
            ', kdim={kdim}, vdim={vdim}, bias={bias}, batch_first={batch_first}, '
            'need_weights={need_weights}'
        )
        if self.name is not None:
            s += ', name = \'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        bias_init = tlx.nn.initializers.zeros()
        weight_init = tlx.nn.initializers.XavierNormal()
        self.q_proj_weight = self._get_weights(
            'q_weight', shape=(self.embed_dim, self.embed_dim), init=weight_init, order=True
        )
        self.k_proj_weight = self._get_weights(
            'k_weight', shape=(self.embed_dim, self.kdim), init=weight_init, order=True
        )
        self.v_proj_weight = self._get_weights(
            'v_weight', shape=(self.embed_dim, self.vdim), init=weight_init, order=True
        )
        self.out_proj_weight = self._get_weights(
            'out_weight', shape=(self.embed_dim, self.embed_dim), init=weight_init, order=True
        )
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.out_bias = None
        if self.bias:
            self.q_bias = self._get_weights('q_bias', shape=(self.embed_dim, ), init=bias_init, order=True)
            self.k_bias = self._get_weights('k_bias', shape=(self.embed_dim, ), init=bias_init, order=True)
            self.v_bias = self._get_weights('v_bias', shape=(self.embed_dim, ), init=bias_init, order=True)
            self.out_bias = self._get_weights('out_bias', shape=(self.embed_dim, ), init=bias_init, order=True)

        self.multiheadattention = tlx.ops.multiheadattention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout, batch_first=self.batch_first,
            need_weights=self.need_weights, q_weight=self.q_proj_weight, k_weight=self.k_proj_weight,
            v_weight=self.v_proj_weight, out_weight=self.out_proj_weight, q_bias=self.q_bias, k_bias=self.k_bias,
            v_bias=self.v_bias, out_bias=self.out_bias, train=self.is_train
        )

    def forward(self, q, k=None, v=None, attn_mask=None, key_padding_mask=None):
        """

        Parameters
        ----------
        q: Tensor
            The queries for multi-head attention. If `batch_first` is ``True``, it is a tensor with shape `[batch_size, query_length, embed_dim]`.
            If `batch_first` is ``False``, it is a tensor with shape `[query_length, batch_size, embed_dim]`. The data type should be float32 or float64.
        k: Tensor
            The keys for multi-head attention. It is a tensor with shape `[batch_size, key_length, kdim]`.
            If `batch_first` is ``False``, it is a tensor with shape `[key_length, batch_size, kdim]`.
            The data type should be float32 or float64. If None, use `query` as `key`. Default is `None`.
        v: Tensor
            The values for multi-head attention. It is a tensor with shape `[batch_size, value_length, vdim]`.
            If `batch_first` is ``False``, it is a tensor with shape `[value_length, batch_size, vdim]`.
            The data type should be float32 or float64. If None, use `value` as `key`. Default is `None`.
        attn_mask: Tensor
            2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            if a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)`. Where N is the batch size, L is the target sequence length, S is the source sequence length.
            ``attn_mask`` ensure that position i is allowed to attend the unmasked positions.
            If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will be unchanged.
            If a BoolTensor is provided, positions with ``True`` is not allowed to attend while ``False`` values will be unchanged.
            If a FloatTensor is provided, it will be added to the attention weight.
        key_padding_mask: Tensor
            if provided, specified padding elements in the key will be ignored by the attention.
            When given a binary mask and a value is True, the corresponding value on the attention layer will be ignored.
            When given a byte mask and a value is non-zero, the corresponding value on the attention layer will be ignored
            :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position with the zero positions will be unchanged.
            If a BoolTensor is provided, the positions with the value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

        Returns
        -------
        attn_output:Tensor
            :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        attn_output_weights:
            :math:`(N, L, S)` where N is the batch size, L is the target sequence length, S is the source sequence length.
        """

        attn_output, attn_output_weights = self.multiheadattention(q, k, v, attn_mask, key_padding_mask)

        if not self._nodes_fixed and self._build_graph:
            self._add_node([q, k, v, attn_mask, key_padding_mask], [attn_output, attn_output_weights])
            self._nodes_fixed = True
        return attn_output, attn_output_weights


class Transformer(Module):
    """A transformer model. User is able to modify the attributes as needed.

    Parameters
    ----------
    d_model: int
        the number of expected features in the encoder/decoder inputs.
    nhead: int
        the number of heads in the multiheadattention model.
    num_encoder_layers:
        the number of sub-encoder-layers in the encoder.
    num_decoder_layers:
        the number of sub-decoder-layers in the decoder.
    dim_feedforward: int
        the dimension of the feedforward network model.
    dropout : float
        a Dropout layer on attn_output_weights. Default: 0.0.
    act: str
        the activation function of encoder/decoder intermediate layer, 'relu' or 'gelu'. Default: 'relu'.
    custom_encoder: Module or None
        custom encoder.
    custom_decoder: Module or None
        custom decoder
    layer_norm_eps: float
        the eps value in layer normalization components. Default: 1e-5.
    batch_first: bool
        If ``True``, then the input and output tensors are provided as `[batch, seq, feature]`. Default: ``False`` `[seq, batch, feature]`.


    Examples
    ---------
    With TensorLayerX

    >>> src = tlx.nn.Input(shape=(4,2,128))
    >>> tgt = tlx.nn.Input(shape=(4,2,128))
    >>> layer = Transformer(d_model=128, nhead=4)
    >>> output = layer(src, tgt)


    References
    ----------
    - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`__
    - `BERT <https://arxiv.org/abs/1810.04805>`__

    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        act='relu',
        custom_encoder=None,
        custom_decoder=None,
        layer_norm_eps=1e-5,
        batch_first=False,
    ):
        super(Transformer, self).__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, act=act,
                layer_norm_eps=layer_norm_eps, batch_first=batch_first
            )
            encoder_norm = tlx.nn.layers.LayerNorm(d_model, epsilon=layer_norm_eps)
            self.encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, act=act,
                layer_norm_eps=layer_norm_eps, batch_first=batch_first
            )
            decoder_norm = tlx.nn.layers.LayerNorm(d_model, epsilon=layer_norm_eps)
            self.decoder = TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=decoder_norm
            )

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(
        self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
        tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):
        """

        Parameters
        ----------
        src: Tensor
            the sequence to the encoder.
        tgt: Tensor
            the sequence to the decoder.
        src_mask: Tensor
            the additive mask for the src sequence.
        tgt_mask: Tensor
            the additive mask for the tgt sequence.
        memory_mask: Tensor
            the additive mask for the encoder output.
        src_key_padding_mask: Tensor
            mask for src keys per batch.
        tgt_key_padding_mask: Tensor
            mask for tgt keys per batch.
        memory_key_padding_mask: Tensor
            mask for memory keys per batch.


        """
        if not self.batch_first and src.shape[1] != tgt.shape[1]:
            raise ValueError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.shape[0] != tgt.shape[0]:
            raise ValueError("the batch number of src and tgt must be equal")

        if src.shape[2] != self.d_model or tgt.shape[2] != self.d_model:
            raise ValueError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        if not self._nodes_fixed and self._build_graph:
            self._add_node([src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask], output)
            self._nodes_fixed = True
        return output

    def generate_square_subsequent_mask(self, length):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).

        Parameters
        ----------
        length: int
            The length of sequence.

        Examples
        ---------
        With TensorLayerX

        >>> length = 5
        >>> mask = transformer.generate_square_subsequent_mask(length)
        >>> print(mask)
        >>> [[  0. -inf -inf -inf -inf]
        >>> [  0.   0. -inf -inf -inf]
        >>> [  0.   0.   0. -inf -inf]
        >>> [  0.   0.   0.   0. -inf]
        >>> [  0.   0.   0.   0.   0.]]

        """
        return tlx.triu(tlx.ones(shape=(length, length)) * -np.inf, 1)


class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers

    Parameters
    ----------
    encoder_layer: Module
        an instance of the TransformerEncoderLayer() class.
    num_layers : int
        the number of sub-encoder-layers in the encoder.
    norm: None
        the layer normalization component.

    Examples
    ---------
    With TensorLayerX

    >>> q = tlx.nn.Input(shape=(4,2,128))
    >>> attn_mask = tlx.convert_to_tensor(np.zeros((4,4)),dtype='bool')
    >>> encoder = TransformerEncoderLayer(128, 2, 256)
    >>> encoder = TransformerEncoder(encoder, num_layers=3)
    >>> output = encoder(q, mask=attn_mask)


    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        # self.encoder_layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.encoder_layers = ModuleList(
            [(encoder_layer if i == 0 else type(encoder_layer)(**encoder_layer._config)) for i in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """

        Parameters
        ----------
        src: Tensor
            the sequence to the encoder.
        mask: Tensor
            the mask for the src sequence.
        src_key_padding_mask:
            the mask for the src keys per batch.

        """
        output = src
        for module in self.encoder_layers:
            output = module(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        if not self._nodes_fixed and self._build_graph:
            self._add_node([src, mask, src_key_padding_mask], output)
            self._nodes_fixed = True
        return output


class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers

    Parameters
    ----------
    decoder_layer: Module
        an instance of the TransformerDecoderLayer() class.
    num_layers : int
        the number of sub-decoder-layers in the decoder.
    norm: None
        the layer normalization component.

    Examples
    ---------
    With TensorLayerX

    >>> q = tlx.nn.Input(shape=(4,2,128))
    >>> decoder = TransformerDecoderLayer(128, 2, 256)
    >>> decoder = TransformerDecoder(decoder, num_layers=3)
    >>> output = decoder(q, q)


    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = ModuleList(
            [(decoder_layer if i == 0 else type(decoder_layer)(**decoder_layer._config)) for i in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):
        """

        Parameters
        ----------
        tgt: Tensor
            the sequence to the decoder.
        memory: Tensor
            the sequence from the last layer of the encoder.
        tgt_mask: Tensor
            the mask for the tgt sequence.
        memory_mask: Tensor
            the mask for the memory sequence.
        tgt_key_padding_mask: Tensor
            the mask for the tgt keys per batch.
        memory_key_padding_mask: Tensor
            the mask for the memory keys per batch.

        """

        output = tgt

        for module in self.decoder_layers:
            output = module(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        if not self._nodes_fixed and self._build_graph:
            self._add_node([tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask], output)
            self._nodes_fixed = True
        return output


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".

    Parameters
    ----------
    d_model: int
        total dimension of the model.
    nhead : int
        The number of heads in multi-head attention.
    dim_feedforward:int
        the dimension of the feedforward network model.
    dropout : float
        a Dropout layer on attn_output_weights. Default: 0.1.
    act: str
        The activation function in the feedforward network. 'relu' or 'gelu'. Default 'relu'.
    layer_norm_eps: float
         the eps value in layer normalization components. Default 1e-5.
    batch_first: bool
        If ``True``, then the input and output tensors are provided as `[batch, seq, feature]`. Default: ``False`` `[seq, batch, feature]`.


    Examples
    ---------
    With TensorLayerX

    >>> q = tlx.nn.Input(shape=(4,2,128))
    >>> attn_mask = tlx.convert_to_tensor(np.zeros((4,4)),dtype='bool')
    >>> encoder = TransformerEncoderLayer(128, 2, 256)
    >>> output = encoder(q, src_mask=attn_mask)


    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        act='relu',
        layer_norm_eps=1e-5,
        batch_first=False,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = tlx.nn.layers.Linear(in_features=d_model, out_features=dim_feedforward)
        self.dropout1 = tlx.nn.layers.Dropout(float(dropout))
        self.linear2 = tlx.nn.layers.Linear(in_features=dim_feedforward, out_features=d_model)

        self.norm1 = tlx.nn.layers.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.norm2 = tlx.nn.layers.LayerNorm(d_model, epsilon=layer_norm_eps)

        self.dropout2 = tlx.nn.layers.Dropout(float(dropout))
        self.dropout3 = tlx.nn.layers.Dropout(float(dropout))
        if act == 'relu':
            self.act = tlx.relu
        elif act == 'gelu':
            self.act = tlx.gelu
        else:
            raise ValueError("activation should be relu or gelu, but got {}".format(act))

        logging.info(
            "TransformerEncoderLayer %s: d_model: %d nhead: %d dim_feedforward: %d dropout: %f act: %s" % (
                self.name, d_model, nhead, dim_feedforward, dropout,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """

        Parameters
        ----------
        src: Tensor
            the sequence to the encoder layer.
        src_mask: Tensor or None
            the mask for the src sequence.
        src_key_padding_mask: Tensor or None
            the mask for the src keys per batch.

        """

        inputs = [src, src_mask, src_key_padding_mask]

        src1 = self.self_attn(src, src, src, src_mask, src_key_padding_mask)[0]
        src = src + self.dropout1(src1)
        src = self.norm1(src)
        src1 = self.linear2(self.dropout2(self.act(self.linear1(src))))
        src = src + self.dropout3(src1)
        src = self.norm2(src)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, src)
            self._nodes_fixed = True
        return src


class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".

    Parameters
    ----------
    d_model: int
        total dimension of the model.
    nhead : int
        The number of heads in multi-head attention.
    dim_feedforward:int
        the dimension of the feedforward network model.
    dropout : float
        a Dropout layer on attn_output_weights. Default: 0.1.
    act: str
        The activation function in the feedforward network. 'relu' or 'gelu'. Default 'relu'.
    layer_norm_eps: float
         the eps value in layer normalization components. Default 1e-5.
    batch_first: bool
        If ``True``, then the input and output tensors are provided as `[batch, seq, feature]`. Default: ``False`` `[seq, batch, feature]`.


    Examples
    ---------
    With TensorLayerX

    >>> q = tlx.nn.Input(shape=(4,2,128))
    >>> encoder = TransformerDecoderLayer(128, 2, 256)
    >>> output = encoder(q, q)


    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        act='relu',
        layer_norm_eps=1e-5,
        batch_first=False,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout1 = tlx.nn.layers.Dropout(float(dropout))
        self.dropout2 = tlx.nn.layers.Dropout(float(dropout))
        self.dropout3 = tlx.nn.layers.Dropout(float(dropout))
        self.norm1 = tlx.nn.layers.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.norm2 = tlx.nn.layers.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.norm3 = tlx.nn.layers.LayerNorm(d_model, epsilon=layer_norm_eps)
        self.linear1 = tlx.nn.layers.Linear(in_features=d_model, out_features=dim_feedforward)
        self.linear2 = tlx.nn.layers.Linear(in_features=dim_feedforward, out_features=d_model)

        if act == 'relu':
            self.act = tlx.relu
        elif act == 'gelu':
            self.act = tlx.gelu

        logging.info(
            "TransformerDecoderLayer %s: d_model: %d nhead: %d dim_feedforward: %d dropout: %f act: %s" % (
                self.name, d_model, nhead, dim_feedforward, dropout,
                self.act.__class__.__name__ if self.act is not None else 'No Activation'
            )
        )

    def forward(
        self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None
    ):
        """

        Parameters
        ----------
        tgt: Tensor
            the sequence to the decoder layer.
        memory:
            the sequence from the last layer of the encoder.
        tgt_mask:
            the mask for the tgt sequence.
        memory_mask:
            the mask for the memory sequence.
        tgt_key_padding_mask:
            the mask for the tgt keys per batch.
        memory_key_padding_mask:
            the mask for the memory keys per batch.

        """
        inputs = [tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask]

        tgt1 = self.self_attn(tgt, tgt, tgt, tgt_mask, tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt1)
        tgt = self.norm1(tgt)
        tgt1 = self.cross_attn(tgt, memory, memory, memory_mask, memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt1)
        tgt = self.norm2(tgt)
        tgt1 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt1)
        tgt = self.norm3(tgt)

        if not self._nodes_fixed and self._build_graph:
            self._add_node(inputs, tgt)
            self._nodes_fixed = True
        return tgt

