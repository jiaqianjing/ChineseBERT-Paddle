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
# here put the import lib
import json
import os
from typing import List
from pypinyin import pinyin, Style

import paddle
from ..bert.tokenizer import BertTokenizer
__all__ = ['BertDataset']

class BertDataset(object):
    def __init__(self, bert_path, max_length: int = 512):
        super().__init__()
        vocab_file = os.path.join(bert_path, 'vocab.txt')
        config_path = os.path.join(bert_path, 'config')
        self.max_length = max_length
        self.tokenizer = BertTokenizer(vocab_file)
        
        # load pinyin map dict
        with open(os.path.join(config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)
    
    def tokenize_sentence(self, sentence):
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output['input_ids']
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence)
        # assert，token nums should be same as pinyin token nums
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        
        # convert list to tensor
        input_ids = paddle.to_tensor(bert_tokens)
        pinyin_ids = paddle.to_tensor(pinyin_tokens).reshape([-1])
        return input_ids, pinyin_ids

    def convert_sentence_to_pinyin_ids(self, sentence: str) -> List[List[int]]:
        # get offsets
        bert_tokens_offsets = self.tokenizer.get_offset_mapping(sentence)
        bert_tokens_offsets.insert(0, (0, 0))
        bert_tokens_offsets.append((0, 0))
        
        # get tokens
        bert_tokens_tokens = self.tokenizer.tokenize(sentence)
        bert_tokens_tokens.insert(0, '[CLS]')
        bert_tokens_tokens.append('[SEP]')
        
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
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
        for idx, (token, offset) in enumerate(zip(bert_tokens_tokens, bert_tokens_offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids

if __name__ == "__main__":
    sentence = '我喜欢猫'
    CHINESEBERT_PATH='../pretrain_models/torch/ChineseBERT-base/'
    tokenizer = BertDataset(CHINESEBERT_PATH)
    input_ids, pinyin_ids = tokenizer.tokenize_sentence(sentence)
    print("============================== paddle =============================")
    print(f"raw input: {sentence}")
    print(f"input_ids: {input_ids.cpu().detach().numpy()}\npinyin_ids: {pinyin_ids.cpu().detach().numpy()}")
