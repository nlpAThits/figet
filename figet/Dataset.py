#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import math
from random import shuffle
import torch
from torch.autograd import Variable

import figet

from tqdm import tqdm

log = figet.utils.get_logging()


class Dataset(object):

    def __init__(self, data, args, type_quantity, volatile=False):
        self.args = args
        self.type_quantity = type_quantity

        self.volatile = volatile
        self.buckets = {}
        for mention in data:
            type_amount = mention.type_len()
            if type_amount in self.buckets:
                self.buckets[type_amount].append(mention)
            else:
                self.buckets[type_amount] = [mention]

    def to_matrix(self, vocabs, word2vec, args):
        self.matrixes = {}
        for type_len, mentions in self.buckets.items():
            self.matrixes[type_len] = self._bucket_to_matrix(mentions, type_len, vocabs, word2vec, args)
        del self.buckets

    def _bucket_to_matrix(self, mentions, type_len, vocabs, word2vec, args):
        """
        create 4 tensors: mentions, types, lCtx and rCtx

        Used only on PREPROCESSING time
        """
        mention_tensor = torch.Tensor(len(mentions), self.args.emb_size)
        previous_ctx_tensor = torch.LongTensor(len(mentions), args.context_length).fill_(figet.Constants.PAD)
        next_ctx_tensor = torch.LongTensor(len(mentions), args.context_length).fill_(figet.Constants.PAD)
        type_tensor = torch.LongTensor(len(mentions), type_len)

        bar = tqdm(desc="to_matrix_{}".format(type_len), total=len(mentions))

        for i in range(len(mentions)):
            bar.update()
            item = mentions[i]
            item.preprocess(vocabs, word2vec, args)

            mention_tensor[i].narrow(0, 0, item.mention.size(0)).copy_(item.mention)
            type_tensor[i].narrow(0, 0, item.types.size(0)).copy_(item.types)

            if len(item.prev_context.size()) != 0 and item.prev_context.size(0) > 0:
                previous_ctx_tensor[i].narrow(0, 0, item.prev_context.size(0)).copy_(item.prev_context)

            if len(item.next_context.size()) != 0 and item.next_context.size(0) > 0:
                reversed_data = torch.from_numpy(item.next_context.numpy()[::-1].copy())
                next_ctx_tensor[i].narrow(0, args.context_length - item.next_context.size(0), item.next_context.size(0)).copy_(reversed_data)

        bar.close()

        return [mention_tensor.contiguous(), previous_ctx_tensor.contiguous(), next_ctx_tensor.contiguous(), \
               type_tensor.contiguous()]

    def __len__(self):
        try:
            return self.num_batches
        except AttributeError as e:
            raise AttributeError("Dataset.set_batch_size must be invoked before to calculate the length") from e

    def shuffle(self):
        shuffle(self.iteration_order)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size  # 1000
        self.num_batches = 0
        self.iteration_order = []
        for type_len, tensors in self.matrixes.items():
            len_tensor = len(tensors[0])
            bucket_num_batches = math.ceil(len_tensor / batch_size)
            for i in range(bucket_num_batches):
                start_index = batch_size * i
                end_index = batch_size * (i + 1) if batch_size * (i + 1) < len_tensor else len_tensor
                self.iteration_order.append((type_len, start_index, end_index))

            self.num_batches += bucket_num_batches

    def create_one_hot_types(self):
        for key in self.matrixes:
            type_tensor = self.matrixes[key][-1]
            one_hot_vectors = torch.zeros(len(type_tensor), self.type_quantity)
            for i in range(len(type_tensor)):
                indexes = type_tensor[i]
                one_hot_vectors[i][indexes] = 1.0
            self.matrixes[key].append(one_hot_vectors)

    def __getitem__(self, index):
        """
        :param index:
        :return: Matrices of different parts (head string, contexts, types) of every instance
        """
        bucket, start_idx, end_index = self.iteration_order[index]
        batch_indexes = torch.arange(start_idx, end_index, dtype=torch.long)

        return [self.process_batch(tensor, batch_indexes) for tensor in self.matrixes[bucket]]

    def process_batch(self, data_tensor, indexes):
        batch_data = data_tensor[indexes]
        return self.to_cuda(batch_data)

    def to_cuda(self, batch_data):
        if torch.cuda.is_available():
            batch_data = batch_data.cuda()
        return Variable(batch_data, volatile=self.volatile)

    def subsample(self, length=None):
        """
        :param length: of the subset. If None, then length is one batch size, at most.
        :return: shuffled subset of self.
        """
        if not length:
            length = self.batch_size

        other = Dataset([], self.args, self.type_quantity, self.volatile)

        other.matrixes = {}
        for type_len, tensors in self.matrixes.items():
            other.matrixes[type_len] = [tensor[:length] for tensor in tensors]

        other.set_batch_size(self.batch_size)

        return other
