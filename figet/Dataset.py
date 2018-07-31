#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import math
import torch
from torch.autograd import Variable

import figet
from figet.utils import to_sparse
from figet.Constants import TYPE_VOCAB

from tqdm import tqdm

log = figet.utils.get_logging()


class Dataset(object):

    GPUS = False

    def __init__(self, data, args, volatile=False):
        self.data = data        # list of figet.Mentions
        self.args = args

        self.volatile = volatile
        self.cached_out = None

    def __len__(self):
        try:
            return self.num_batches
        except AttributeError:
            log.info("Dataset.set_batch_size must be invoked before to calculate the length")
        # except AttributeError as e:       # PYTHON 3
        #     raise AttributeError("") from e

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size  # 1000
        self.num_batches = math.ceil(len(self.data) / batch_size)

    def shuffle(self):
        self.data = [self.data[i] for i in torch.randperm(len(self.data))]

    def to_matrix(self, vocabs, word2vec, args):
        """
        create 4 tensors: mentions, types, lCtx and rCtx
        """
        mention_tensor = torch.Tensor(len(self.data), self.args.emb_size)
        previous_ctx_tensor = torch.LongTensor(len(self.data), args.context_length).fill_(figet.Constants.PAD)
        next_ctx_tensor = torch.LongTensor(len(self.data), args.context_length).fill_(figet.Constants.PAD)

        self.type_dims = vocabs[TYPE_VOCAB].size()
        type_tensors = []

        bar = tqdm(desc="to_matrix", total=len(self.data))

        for i in xrange(len(self.data)):
            bar.update()
            item = self.data[i]
            item.preprocess(vocabs, word2vec, args)

            mention_tensor[i].narrow(0, 0, item.mention.size(0)).copy_(item.mention)

            if len(item.prev_context.size()) != 0:
                previous_ctx_tensor[i].narrow(0, 0, item.prev_context.size(0)).copy_(item.prev_context)

            if len(item.next_context.size()) != 0:
                reversed_data = torch.from_numpy(item.next_context.numpy()[::-1].copy())
                next_ctx_tensor[i].narrow(0, args.context_length - item.next_context.size(0), item.next_context.size(0)).copy_(reversed_data)

            type_tensors.append(to_sparse(item.types))

            item.clear()

        bar.close()
        self.mention_tensor = mention_tensor.contiguous()
        self.previous_ctx_tensor = previous_ctx_tensor.contiguous()
        self.next_ctx_tensor = next_ctx_tensor.contiguous()
        self.type_tensors = type_tensors

    def __getitem__(self, index):
        """
        :param index:
        :return: Matrices of different parts (head string, context) of every instance
        """
        index = int(index % self.num_batches)
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)    # WTF, this is obvious
        batch_ini = self.batch_size * index
        batch_end = self.batch_size * (index + 1)

        mention_batch = self.process_batch(self.mention_tensor, batch_ini, batch_end)
        previous_ctx_batch = self.process_batch(self.previous_ctx_tensor, batch_ini, batch_end)
        next_ctx_batch = self.process_batch(self.next_ctx_tensor, batch_ini, batch_end)

        type_batch = self.get_type_batch(batch_ini, batch_end)

        return (
            mention_batch,
            (previous_ctx_batch, None),
            (next_ctx_batch, None),
            type_batch, None, None, None
        )

    def get_type_batch(self, batch_ini, batch_end):
        type_batch = []
        for nonzeros in self.type_tensors[batch_ini: batch_end]:
            type_vec = torch.Tensor(self.type_dims).fill_(0)
            for non_zero_idx in nonzeros:
                type_vec[non_zero_idx] = 1
            type_batch.append(type_vec)

        type_batch = torch.stack(type_batch)
        type_batch = type_batch.contiguous()
        return self.to_cuda(type_batch)

    def process_batch(self, data_tensor, ini, end):
        batch_data = data_tensor[ini: end]
        return self.to_cuda(batch_data)

    def to_cuda(self, batch_data):
        if Dataset.GPUS:
            batch_data = batch_data.cuda()
        return Variable(batch_data, volatile=self.volatile)

