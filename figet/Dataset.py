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

    def __init__(self, data, args, volatile=False):
        self.args = args

        self.volatile = volatile
        self.shuffled = False
        self.indexes = None
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

        bar = tqdm(desc="to_matrix", total=len(mentions))

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

        return mention_tensor.contiguous(), previous_ctx_tensor.contiguous(), next_ctx_tensor.contiguous(), \
               type_tensor.contiguous()

    def __len__(self):
        try:
            return self.num_batches
        except AttributeError as e:
            raise AttributeError("Dataset.set_batch_size must be invoked before to calculate the length") from e

    def shuffle(self):
        self.shuffled = True
        self.type_lengths = self.matrixes.keys()
        shuffle(self.type_lengths)


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size  # 1000
        self.num_batches = 0
        for type_len, tensors in self.matrixes:


            # armar una lista que cada indice es el index de get_item y dentro tiene una tupla con (nro_bucket, index_ini; index_end)
            # luego es all igual, o bien los shuffleo o no, y cuando me llega un item solo busco en esa lista lo que haya y devuelvo esos slices de tensores



            len_tensor = len(tensors[0])
            self.num_batches += math.ceil(len_tensor / batch_size)

    def __getitem__(self, index):
        """
        :param index:
        :return: Matrices of different parts (head string, context) of every instance
        """


        # esto deberia ser
        # bucket, start_idx, end_index = self.iteration_order[index]
        # luego hago lo mismo que ahora sobre los tensores del _bucket_ correspondiente



        index = int(index % self.num_batches)
        start_idx = self.batch_size * index
        end_index = self.batch_size * (index + 1) if self.batch_size * (index + 1) < self.len_data else self.len_data
        if self.shuffled:
            batch_indexes = self.indexes[start_idx: end_index]
        else:
            batch_indexes = torch.arange(start_idx, end_index, dtype=torch.long)

        mention_batch = self.process_batch(self.mention_tensor, batch_indexes)
        previous_ctx_batch = self.process_batch(self.previous_ctx_tensor, batch_indexes)
        next_ctx_batch = self.process_batch(self.next_ctx_tensor, batch_indexes)
        type_batch = self.get_type_batch(batch_indexes)

        return mention_batch, previous_ctx_batch, next_ctx_batch, type_batch

    def process_batch(self, data_tensor, indexes):
        batch_data = data_tensor[indexes]
        return self.to_cuda(batch_data)

    def to_cuda(self, batch_data):
        if torch.cuda.is_available():
            batch_data = batch_data.cuda()
        return Variable(batch_data, volatile=self.volatile)

    def get_type_batch(self, batch_indexes):


    # def subsample(self, length=None):
    #     """
    #     CURRENTLY NOT USED
    #     :param length: of the subset. If None, then length is one batch size, at most.
    #     :return: shuffled subset of self.
    #     """
    #     if not length:
    #         length = self.batch_size
    #     if length > self.len_data:
    #         length = self.len_data
    #
    #     other = Dataset(None, self.args, self.volatile)
    #     other.batch_size = self.batch_size
    #     other.num_batches = math.ceil(length / self.batch_size)
    #     other.len_data = length
    #
    #     idx = torch.randperm(len(self.type_tensor))[:length]
    #     other.mention_tensor = self.mention_tensor[idx]
    #     other.previous_ctx_tensor = self.previous_ctx_tensor[idx]
    #     other.next_ctx_tensor = self.next_ctx_tensor[idx]
    #     other.type_tensor = self.type_tensor[idx]
    #
    #     return other
