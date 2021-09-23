# -*- coding: utf-8 -*-
"""
@File    :   run.py
@Time    :   2021/09/11 16:00:04
@Author  :   Jia Qianjing
@Version :   1.0
@Contact :   jiaqianjing@gmail.com
@Desc    :   Non
"""
import sys
sys.path.append("/root/paddlejob/workspace/code/ChineseBERT-Paddle/Paddle_ChineseBert/PaddleNLP")
from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup


parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=13, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args(args=[])

data_prefix="/root/paddlejob/workspace/code"

def read_train_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        head = None
        for no, line in enumerate(f):
            if no == 30001:
                break
            data = line.strip().split('\t', 2)
            if not head:
                head = data
            else:
                sentence1, sentence2, label = data
                yield {"sentence1": sentence1, "sentence2": sentence2, "label": label}

def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        head = None
        for line in f:
            data = line.strip().split('\t')
            if not head:
                head = data
            else:
                lan = data[0]
                label = data[1]
                sentence1 = data[6]
                sentence2 = data[7]
                if lan != 'zh':
                    continue
                else:
                    yield {"sentence1": sentence1, "sentence2": sentence2, "label": label}

def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    # 【FOCUS】 --> https://github.com/ShannonAI/ChineseBert/blob/main/datasets/xnli_dataset.py
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2, "contradictory": 2}
    first, second, third = example['sentence1'], example['sentence2'], example['label']

    first_input_tokens = tokenizer.tokenize(first)
    first_pinyin_tokens = tokenizer.convert_sentence_to_pinyin_ids(first, with_specail_token=False)

    second_input_tokens = tokenizer.tokenize(second)
    second_pinyin_tokens = tokenizer.convert_sentence_to_pinyin_ids(second, with_specail_token=False)

    label = np.array([label_map[third]], dtype="int64")

    # convert sentence to id
    bert_tokens = tokenizer.convert_tokens_to_ids(first_input_tokens) + [102] + tokenizer.convert_tokens_to_ids(second_input_tokens)
    pinyin_tokens = first_pinyin_tokens + [[0] * 8] + second_pinyin_tokens
    if len(bert_tokens) > max_seq_length - 2:
        bert_tokens = bert_tokens[:max_seq_length - 2]
        pinyin_tokens = pinyin_tokens[:max_seq_length - 2]
    # assert
    assert len(bert_tokens) <= max_seq_length
    assert len(bert_tokens) == len(pinyin_tokens)

    input_ids = [101] + bert_tokens + [102]
    pinyin_ids = [[0] * 8] + pinyin_tokens + [[0] * 8]

    input_ids = np.array(input_ids)
    pinyin_ids = np.array(pinyin_ids)


    return input_ids, pinyin_ids, label

def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    # shuffle = False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, pinyin_ids, labels = batch
        logits = model(input_ids, pinyin_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    # ChineseBertModel
    CHINESEBERT_PADDLE_PATH = "./pretrain_models/paddle/ChineseBERT-large"
    model = ppnlp.transformers.GlyceBertForSequenceClassification.from_pretrained(CHINESEBERT_PADDLE_PATH, num_classes=4)

    # ChineseBertTokenizer
    tokenizer = ppnlp.transformers.ChineseBertTokenizer(CHINESEBERT_PADDLE_PATH)

    train_ds = load_dataset(read_train_ds, data_path=f'{data_prefix}/ChineseBERT-Paddle/data/XNLI/xnli.train.tsv', lazy=False)
    dev_ds = load_dataset(read, data_path=f'{data_prefix}/ChineseBERT-Paddle/data/XNLI/xnli.dev.tsv', lazy=False)
    test_ds = load_dataset(read, data_path=f'{data_prefix}/ChineseBERT-Paddle/data/XNLI/xnli.test.tsv', lazy=False)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # pinyin_ids
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]
    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)
    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    test_data_loader = create_dataloader(
        test_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-08,
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, pinyin_ids, labels = batch
            logits = model(input_ids, pinyin_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            if global_step % 100 == 0 and rank == 0:
                # save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                print("dev eval:")
                dev_acc = evaluate(model, criterion, metric, dev_data_loader)
                print("test eval:")
                test_acc = evaluate(model, criterion, metric, test_data_loader)
                # if test_acc >= 81.6:
                #     sys.exit(0)
                # model._layers.save_pretrained(save_dir)
                # tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    do_train()
