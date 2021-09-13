#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   modeling.py
@Time    :   2021/09/11 16:10:29
@Author  :   Jia Qianjing
@Version :   1.0
@Contact :   jiaqianjing@gmail.com
@Desc    :   None
'''

# here put the import lib
# Paddle PinyinEmbedding
import json
import os
from re import I
from typing import List

import numpy as np
import paddle
import paddle.nn as nn
from .. import PretrainedModel, register_base_model

__all__ = [
    'GlyceBertPretrainedModel', 'GlyceBertModel',
    'GlyceBertForSequenceClassification'
]


class PinyinEmbedding(nn.Layer):
    def __init__(self, embedding_size: int, pinyin_out_dim: int, config_path):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()
        with open(os.path.join(config_path, 'pinyin_map.json')) as fin:
            pinyin_dict = json.load(fin)
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(len(pinyin_dict['idx2char']),
                                      embedding_size)
        self.conv = nn.Conv1D(in_channels=embedding_size,
                              out_channels=self.pinyin_out_dim,
                              kernel_size=2,
                              stride=1,
                              padding=0,
                              bias_attr=True)

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(
            pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.reshape(
            (-1, pinyin_locs,
             embed_size))  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.transpose(
            [0, 2, 1])  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(
            input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = nn.functional.max_pool1d(
            pinyin_conv,
            pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.reshape(
            (bs, sentence_length,
             self.pinyin_out_dim))  # [bs,sentence_length,pinyin_out_dim]


class GlyphEmbedding(nn.Layer):
    """Glyph2Image Embedding"""
    def __init__(self, font_npy_files: List[str]):
        super(GlyphEmbedding, self).__init__()
        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_npy_files
        ]
        self.vocab_size = font_arrays[0].shape[0]
        self.font_num = len(font_arrays)
        self.font_size = font_arrays[0].shape[-1]
        # N, C, H, W
        font_array = np.stack(font_arrays, axis=1)
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.font_size**2 *
                                      self.font_num)
        self.embedding.weight.set_value(
            font_array.reshape([self.vocab_size, -1]))

    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)


class FusionBertEmbeddings(nn.Layer):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """
    def __init__(self, name_or_path, vocab_size, hidden_size,
                 max_position_embeddings, type_vocab_size, layer_norm_eps,
                 hidden_dropout_prob):
        super(FusionBertEmbeddings, self).__init__()
        config_path = os.path.join(name_or_path, 'config')
        font_files = []
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(embedding_size=128,
                                                 pinyin_out_dim=hidden_size,
                                                 config_path=config_path)
        self.glyph_embeddings = GlyphEmbedding(font_npy_files=font_files)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.glyph_map = nn.Linear(1728, hidden_size, bias_attr=True)
        self.map_fc = nn.Linear(hidden_size * 3, hidden_size, bias_attr=True)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            paddle.arange(max_position_embeddings).expand((1, -1)))

    def forward(self,
                input_ids=None,
                pinyin_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        pinyin_embeddings = self.pinyin_embeddings(
            pinyin_ids)  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_map(
            self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]
        # fusion layer
        concat_embeddings = paddle.concat(
            (word_embeddings, pinyin_embeddings, glyph_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(paddle.nn.Layer):
    """
    Pool the result of BertEncoder.
    """
    def __init__(self, hidden_size, pool_act="tanh"):
        super(BertPooler, self).__init__()
        self.dense = paddle.nn.Linear(hidden_size, hidden_size)
        self.activation = paddle.nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class GlyceBertPretrainedModel(PretrainedModel):
    model_config_file = "model_config.json"
    pretrained_init_configuration = {}
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {}
    base_model_prefix = "bert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(mean=0.0,
                                         std=self.initializer_range,
                                         shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class GlyceBertModel(GlyceBertPretrainedModel):
    r"""
    PaddleGlyceBertModel
    """
    def __init__(self,
                 name_or_path,
                 vocab_size=23236,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pad_token_id=0,
                 pool_act="tanh",
                 layer_norm_eps=1e-12):
        super(GlyceBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = FusionBertEmbeddings(name_or_path, vocab_size,
                                               hidden_size,
                                               max_position_embeddings,
                                               type_vocab_size, layer_norm_eps,
                                               hidden_dropout_prob)
        encoder_layer = paddle.nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = paddle.nn.TransformerEncoder(encoder_layer,
                                                    num_hidden_layers)
        self.pooler = BertPooler(hidden_size, pool_act=pool_act)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids=None,
        pinyin_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id).astype(
                    self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(input_ids=input_ids,
                                           pinyin_ids=pinyin_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)

        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class GlyceBertForSequenceClassification(GlyceBertPretrainedModel):
    """
    GlyceBert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.

    Args:
        bert (:class:`GlyceBertModel`):
            An instance of GlyceBertModel.
        num_classes (int, optional):
            The number of classes. Defaults to `2`.
        dropout (float, optional):
            The dropout probability for output of GlyceBertModel.
            If None, use the same value as `hidden_dropout_prob` of `GlyceBertModel`
            instance `bert`. Defaults to None.
    """
    def __init__(self, bert, num_classes=2, dropout=None):
        super(GlyceBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else self.bert.
                                  config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.bert.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                pinyin_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        r"""
        The GlyceBertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids(Tensor, optional):
                See :class:`BertModel`.
            attention_mask (list, optional):
                See :class:`BertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

        """

        _, pooled_output = self.bert(input_ids,
                                     pinyin_ids,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
