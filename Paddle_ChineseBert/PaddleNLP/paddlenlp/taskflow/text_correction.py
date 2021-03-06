# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import math
import os
import copy
import itertools

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..transformers import ErnieTokenizer, ErnieModel
from ..transformers import is_chinese_char
from ..datasets import load_dataset
from ..data import Stack, Pad, Tuple, Vocab
from .utils import download_file, add_docstrings, static_mode_guard, dygraph_mode_guard
from .models import ErnieForCSC
from .task import Task

usage = r"""
           from paddlenlp import Taskflow

           text_correction = Taskflow("text_correction")
           text_correction('遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。')
           '''
           [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
             'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
             'errors': [{'position': 3, 'correction': {'竟': '境'}}]}
           ]
           '''

           text_correction(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。',
                            '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
           '''
           [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 
             'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 
             'errors': [{'position': 3, 'correction': {'竟': '境'}}]}, 
            {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 
             'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。', 
             'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}
           ]
           '''

         """

URLS = {
    "csc-ernie-1.0": [
        "https://paddlenlp.bj.bcebos.com/taskflow/text_correction/csc-ernie-1.0/csc-ernie-1.0.pdparams",  # model url
        None,  # md5
        "https://paddlenlp.bj.bcebos.com/taskflow/text_correction/csc-ernie-1.0/pinyin_vocab.txt",  # pinyin vocab url
    ],
}

TASK_MODEL_MAP = {"csc-ernie-1.0": "ernie-1.0"}


class CSCTask(Task):
    """
    The text generation model to predict the question or chinese  poetry. 
    Args:
        task(string): The name of task.
        model(string): The model name in the task.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 
    """

    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)
        self._static_mode = True
        self._usage = usage
        # Download pinyin vocab
        pinyin_vocab_path = download_file(self._task_path, "pinyin_vocab.txt",
                                          URLS[self.model][2])
        self._pinyin_vocab = Vocab.load_vocabulary(
            pinyin_vocab_path, unk_token='[UNK]', pad_token='[PAD]')

        if self._static_mode:
            download_file(self._task_path,
                          "static" + os.path.sep + "inference.pdiparams",
                          URLS[self.model][0], URLS[self.model][1])
            self._get_inference_model()
        else:
            self._construct_model(model)
        self._construct_tokenizer(model)
        try:
            import pypinyin
        except:
            raise ImportError(
                "Please install the dependencies first, pip install pypinyin --upgrade"
            )
        self._pypinyin = pypinyin
        self._max_seq_length = 128

    def _construct_input_spec(self):
        """
       Construct the input spec for the convert dygraph model to static model.
       """
        self._input_spec = [
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='input_ids'),
            paddle.static.InputSpec(
                shape=[None, None], dtype="int64", name='pinyin_ids'),
        ]

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        ernie = ErnieModel.from_pretrained(TASK_MODEL_MAP[model])
        model_instance = ErnieForCSC(
            ernie,
            pinyin_vocab_size=len(self._pinyin_vocab),
            pad_pinyin_id=self._pinyin_vocab[self._pinyin_vocab.pad_token])
        # Load the model parameter for the predict
        model_path = download_file(self._task_path, model + ".pdparams",
                                   URLS[model][0], URLS[model][1])
        state_dict = paddle.load(model_path)
        model_instance.set_state_dict(state_dict)
        model_instance.eval()
        self._model = model_instance

    def _construct_tokenizer(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self._tokenizer = ErnieTokenizer.from_pretrained(TASK_MODEL_MAP[model])

    def _preprocess(self, inputs, padding=True, add_special_tokens=True):
        inputs = self._check_input_text(inputs)
        batch_size = self.kwargs[
            'batch_size'] if 'batch_size' in self.kwargs else 1
        trans_func = self._convert_example

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self._tokenizer.pad_token_id),  # input
            Pad(axis=0, pad_val=self._tokenizer.pad_token_type_id),  # segment
            Pad(axis=0, pad_val=self._pinyin_vocab.token_to_idx[self._pinyin_vocab.pad_token]),  # pinyin
            Stack(axis=0, dtype='int64'),  # length
        ): [data for data in fn(samples)]

        examples = []
        texts = []
        for text in inputs:
            if not (isinstance(text, str) and len(text) > 0):
                continue
            example = {"source": text.strip()}
            input_ids, token_type_ids, pinyin_ids, length = trans_func(example)
            examples.append((input_ids, token_type_ids, pinyin_ids, length))
            texts.append(example["source"])

        batch_examples = [
            examples[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]
        batch_texts = [
            texts[idx:idx + batch_size]
            for idx in range(0, len(examples), batch_size)
        ]
        outputs = {}
        outputs['batch_examples'] = batch_examples
        outputs['batch_texts'] = batch_texts
        self.batchify_fn = batchify_fn
        return outputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_tokenize` function. 
        """
        results = []
        with static_mode_guard():
            for examples in inputs['batch_examples']:
                token_ids, token_type_ids, pinyin_ids, lengths = self.batchify_fn(
                    examples)
                self.input_handles[0].copy_from_cpu(token_ids)
                self.input_handles[1].copy_from_cpu(pinyin_ids)
                self.predictor.run()
                det_preds = self.output_handle[0].copy_to_cpu()
                char_preds = self.output_handle[1].copy_to_cpu()

                batch_result = []
                for i in range(len(lengths)):
                    batch_result.append(
                        (det_preds[i], char_preds[i], lengths[i]))
                results.append(batch_result)
        inputs['batch_results'] = results
        return inputs

    def _postprocess(self, inputs):
        """
        The model output is the logits and probs, this function will convert the model output to raw text.
        """
        final_results = []

        for examples, texts, results in zip(inputs['batch_examples'],
                                            inputs['batch_texts'],
                                            inputs['batch_results']):
            for i in range(len(examples)):
                result = {}
                det_pred, char_preds, length = results[i]
                pred_result = self._parse_decode(texts[i], char_preds, det_pred,
                                                 length)
                result['source'] = texts[i]
                result['target'] = ''.join(pred_result)
                errors_result = []
                for i, (
                        source_token, target_token
                ) in enumerate(zip(result['source'], result['target'])):
                    if source_token != target_token:
                        errors_result.append({
                            'position': i,
                            'correction': {
                                source_token: target_token
                            }
                        })
                result['errors'] = errors_result
                final_results.append(result)
        return final_results

    def _convert_example(self, example):
        source = example["source"]
        words = self._tokenizer.tokenize(text=source)
        if len(words) > self._max_seq_length - 2:
            words = words[:self._max_seq_length - 2]
        length = len(words)
        words = ['[CLS]'] + words + ['[SEP]']
        input_ids = self._tokenizer.convert_tokens_to_ids(words)
        token_type_ids = [0] * len(input_ids)

        # Use pad token in pinyin emb to map word emb [CLS], [SEP]
        pinyins = self._pypinyin.lazy_pinyin(
            source,
            style=self._pypinyin.Style.TONE3,
            neutral_tone_with_five=True)

        pinyin_ids = [0]
        # Align pinyin and chinese char
        pinyin_offset = 0
        for i, word in enumerate(words[1:-1]):
            pinyin = '[UNK]' if word != '[PAD]' else '[PAD]'
            if len(word) == 1 and is_chinese_char(ord(word)):
                while pinyin_offset < len(pinyins):
                    current_pinyin = pinyins[pinyin_offset][:-1]
                    pinyin_offset += 1
                    if current_pinyin in self._pinyin_vocab:
                        pinyin = current_pinyin
                        break
            pinyin_ids.append(self._pinyin_vocab[pinyin])

        pinyin_ids.append(0)
        assert len(input_ids) == len(
            pinyin_ids
        ), "length of input_ids must be equal to length of pinyin_ids"
        return input_ids, token_type_ids, pinyin_ids, length

    def _parse_decode(self, words, corr_preds, det_preds, lengths):
        UNK = self._tokenizer.unk_token
        UNK_id = self._tokenizer.convert_tokens_to_ids(UNK)
        tokens = self._tokenizer.tokenize(words)
        if len(tokens) > self._max_seq_length - 2:
            tokens = tokens[:self._max_seq_length - 2]
        corr_pred = corr_preds[1:1 + lengths].tolist()
        det_pred = det_preds[1:1 + lengths].tolist()
        words = list(words)
        if len(words) > self._max_seq_length - 2:
            words = words[:self._max_seq_length - 2]

        assert len(tokens) == len(
            corr_pred
        ), "The number of tokens should be equal to the number of labels {}: {}: {}".format(
            len(tokens), len(corr_pred), tokens)
        pred_result = ""

        align_offset = 0
        # Need to be aligned
        if len(words) != len(tokens):
            first_unk_flag = True
            for j, word in enumerate(words):
                if word.isspace():
                    tokens.insert(j + 1, word)
                    corr_pred.insert(j + 1, UNK_id)
                    det_pred.insert(j + 1, 0)  # No error
                elif tokens[j] != word:
                    if self._tokenizer.convert_tokens_to_ids(word) == UNK_id:
                        if first_unk_flag:
                            first_unk_flag = False
                            corr_pred[j] = UNK_id
                            det_pred[j] = 0
                        else:
                            tokens.insert(j, UNK)
                            corr_pred.insert(j, UNK_id)
                            det_pred.insert(j, 0)  # No error
                        continue
                    elif tokens[j] == UNK:
                        # Remove rest unk
                        k = 0
                        while k + j < len(tokens) and tokens[k + j] == UNK:
                            k += 1
                        tokens = tokens[:j] + tokens[j + k:]
                        corr_pred = corr_pred[:j] + corr_pred[j + k:]
                        det_pred = det_pred[:j] + det_pred[j + k:]
                    else:
                        # Maybe English, number, or suffix
                        token = tokens[j].lstrip("##")
                        corr_pred = corr_pred[:j] + [UNK_id] * len(
                            token) + corr_pred[j + 1:]
                        det_pred = det_pred[:j] + [0] * len(token) + det_pred[
                            j + 1:]
                        tokens = tokens[:j] + list(token) + tokens[j + 1:]
                first_unk_flag = True

        for j, word in enumerate(words):
            candidates = self._tokenizer.convert_ids_to_tokens(corr_pred[j])
            if det_pred[j] == 0 or candidates == UNK or candidates == '[PAD]':
                pred_result += word
            else:
                pred_result += candidates.lstrip("##")

        return pred_result
