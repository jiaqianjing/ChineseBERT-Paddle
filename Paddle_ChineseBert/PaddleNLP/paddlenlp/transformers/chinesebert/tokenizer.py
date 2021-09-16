#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tokenzier.py
@Time    :   2021/09/11 16:00:04
@Author  :   Jia Qianjing
@Version :   1.0
@Contact :   jiaqianjing@gmail.com
@Desc    :   None
'''
import json
import os
from typing import List

import numpy as np
from pypinyin import Style, pinyin

from .. import BasicTokenizer, PretrainedTokenizer, WordpieceTokenizer

__all__ = ['ChineseBertTokenizer']


class ChineseBertTokenizer(PretrainedTokenizer):
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {}
    pretrained_init_configuration = {}
    padding_side = 'right'

    def __init__(self,
                 bert_path,
                 max_seq_len=512,
                 do_lower_case=True,
                 unk_token="[UNK]",
                 sep_token="[SEP]",
                 pad_token="[PAD]",
                 cls_token="[CLS]",
                 mask_token="[MASK]"):
        vocab_file = os.path.join(bert_path, 'vocab.txt')
        config_path = os.path.join(bert_path, 'config')
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        self.max_seq_len = max_seq_len
        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'),
                  encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'),
                  encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'),
                  encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
                                                      unk_token=unk_token)

    def tokenize_sentence(self, sentence):
        # convert sentence to ids
        tokenizer_output = self.encode(sentence)
        input_ids = tokenizer_output['input_ids']
        pinyin_ids = self.convert_sentence_to_pinyin_ids(sentence)
        # assert，token nums should be same as pinyin token nums
        # assert len(input_ids) <= self.max_seq_len
        # assert len(input_ids) == len(pinyin_ids)

        # convert list to tensor
        # input_ids = paddle.to_tensor(input_ids)
        # pinyin_ids = paddle.to_tensor(pinyin_ids).reshape([-1])

        # convert list to np.array
        input_ids = np.array(input_ids)
        pinyin_ids = np.array(pinyin_ids).reshape([-1, 8])
        return {"input_ids": input_ids, "pinyin_ids": pinyin_ids}

    def convert_sentence_to_pinyin_ids(self, sentence: str, with_specail_token=True) -> List[List[int]]:
        # get offsets
        bert_tokens_offsets = self.get_offset_mapping(sentence)
        if with_specail_token:
            bert_tokens_offsets.insert(0, (0, 0))
            bert_tokens_offsets.append((0, 0))

        # get tokens
        bert_tokens_tokens = self.tokenize(sentence)
        if with_specail_token:
            bert_tokens_tokens.insert(0, '[CLS]')
            bert_tokens_tokens.append('[SEP]')

        # get pinyin of a sentence
        pinyin_list = pinyin(sentence,
                             style=Style.TONE3,
                             heteronym=True,
                             errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(
                zip(bert_tokens_tokens, bert_tokens_offsets)):
            if offset[1] - offset[0] != 1:
                # 非单个字的token，以及 [CLS] [SEP] 特殊 token
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                # 单个字为token且有拼音tensor
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                # 单个字为token但无拼音tensor
                pinyin_ids.append([0] * 8)

        return pinyin_ids

    def convert_tokens_to_pinyin_ids(self,
                                     tokens: List[str]) -> List[List[int]]:
        """
        Example :
            tokens:  ['[CLS]', '你', '多', '大', '了', '？', '[SEP]', '我', '10', '岁', '了', '。', '[SEP]']
        """
        pinyin_ids = []
        for token in tokens:
            if token == '[CLS]' or token == '[SEP]':
                # [CLS]、[SEP] 的 token
                pinyin_ids.append([0] * 8)
                continue
            offset = self.get_offset_mapping(token)[0]
            if offset[1] - offset[0] != 1:
                # 非单个字组成的 token
                pinyin_ids.append([0] * 8)
                continue
            pinyin_string = pinyin(token,
                                   style=Style.TONE3,
                                   heteronym=True,
                                   errors=lambda x: [['not chinese']
                                                     for _ in x])[0][0]
            if pinyin_string == "not chinese":
                # 不是中文
                pinyin_ids.append([0] * 8)
            else:
                if pinyin_string in self.pinyin2tensor:
                    pinyin_ids.append(self.pinyin2tensor[pinyin_string])
                else:
                    ids = [0] * 8
                    for i, p in enumerate(pinyin_string):
                        if p not in self.pinyin_dict["char2idx"]:
                            ids = [0] * 8
                            break
                        ids[i] = self.pinyin_dict["char2idx"][p]
                    pinyin_ids.append(ids)
        return pinyin_ids

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """

        return len(self.vocab)

    def _tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def tokenize(self, text):
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(
                token_ids_0, token_ids_1 if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1
                    if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
