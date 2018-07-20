#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import math
import torch
from torch.autograd import Variable

import figet
from figet.Constants import TYPE_VOCAB

log = figet.utils.get_logging()


class Dataset(object):

    GPUS = False

    def __init__(self, data, batch_size, args, volatile=False):
        self.data = data        # list of figet.Mentions
        self.args = args

        self.batch_size = batch_size    # 1000
        self.num_batches = math.ceil(len(self.data) / batch_size)
        self.volatile = volatile
        self.cached_out = None
        self.mention_tensor = None
        self.gpus = False

    def __len__(self):
        return self.num_batches

    def shuffle(self):
        self.data = [self.data[i] for i in torch.randperm(len(self.data))]

    def to_matrix(self, vocabs, word2vec, args):
        """
        create 4 tensors: mentions, types, lCtx and rCtx
        """
        mention_tensor = torch.Tensor(len(self.data), self.args.emb_size)
        type_tensor = torch.Tensor(len(self.data), vocabs[TYPE_VOCAB].size())
        previous_ctx_tensor = torch.LongTensor(len(self.data), args.context_length).fill_(figet.Constants.PAD)
        next_ctx_tensor = torch.LongTensor(len(self.data), args.context_length).fill_(figet.Constants.PAD)

        for i in xrange(len(self.data)):
            item = self.data[i]
            item.preprocess(vocabs, word2vec, args)

            mention_tensor[i].narrow(0, 0, item.mention.size(0)).copy_(item.mention)
            type_tensor[i].narrow(0, 0, item.types.size(0)).copy_(item.types)

            if len(item.prev_context.size()) != 0:
                previous_ctx_tensor[i].narrow(0, 0, item.prev_context.size(0)).copy_(item.prev_context)

            if len(item.next_context.size()) != 0:
                reversed_data = torch.from_numpy(item.next_context.numpy()[::-1].copy())
                next_ctx_tensor[i].narrow(0, args.context_length - item.next_context.size(0), item.next_context.size(0)).copy_(reversed_data)

            item.clear()

        self.mention_tensor = mention_tensor.contiguous()
        self.type_tensor = type_tensor.contiguous()
        self.previous_ctx_tensor = previous_ctx_tensor.contiguous()
        self.next_ctx_tensor = next_ctx_tensor.contiguous()
    

    def _batchify(self, data, max_length=None, include_lengths=False, reverse=False):
        """
        :param data: list
        :param max_length:
        :param include_lengths:
        :param reverse:
        :return:
            out: data in the shape of a matrix of (len(data) x max_length). It has a row of zeros in the row i if data[i].size() is zero.
                If data[i].size(0) < max_length there will be zeros on the remaining columns.
            out_lengths: list with the length of each row (relevant columns) of the matrix out.
            mask: matrix of the same size than out. Zeros on the rows where out has information, ones otherwise. MASKS ARE NEVER USED!!!!!!!
        """
        if max_length is None:
            lengths = [x.size(0) if len(x.size()) else 0 for x in data]
            # log.debug("Lengths.......................")
            # log.debug(lengths)            For mentions this is always 300, for types is always the same (89 in finet, 824 for me) y it only changes on the features (which are not even used)
            max_length = max(lengths)
        out_lengths = []
        out = data[0].new(len(data), max_length).fill_(figet.Constants.PAD) # matrix full of zeros
        mask = torch.ByteTensor(len(data), max_length).fill_(1)             # matrix full of ones
        for i in xrange(len(data)):
            if len(data[i].size()) == 0:
                out_lengths.append(1)
                continue
            data_length = data[i].size(0)
            out_lengths.append(data_length)
            offset = 0
            if reverse:
                reversed_data = torch.from_numpy(data[i].numpy()[::-1].copy())
                out[i].narrow(0, max_length-data_length, data_length).copy_(reversed_data)
                mask[i].narrow(0, max_length-data_length, data_length).fill_(0)
            else:
                out[i].narrow(0, offset, data_length).copy_(data[i])    # copy data to the matrix
                mask[i].narrow(0, offset, data_length).fill_(0)         # fills mask with zeros
        out = out.contiguous()
        mask = mask.contiguous()
                                        # Esta parte luego cuando se "batchifican"
        if len(self.args.gpus) > 0:
            out = out.cuda()
            mask = mask.cuda()

        out = Variable(out, volatile=self.volatile)
        if include_lengths:
            return out, out_lengths, mask
        return out, None, mask

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
        type_batch = self.process_batch(self.type_tensor, batch_ini, batch_end)
        previous_ctx_batch = self.process_batch(self.previous_ctx_tensor, batch_ini, batch_end)
        next_ctx_batch = self.process_batch(self.next_ctx_tensor, batch_ini, batch_end)

        return (
            mention_batch,
            (previous_ctx_batch, None),
            (next_ctx_batch, None),
            type_batch, None, None, None
        )

    def process_batch(self, data_tensor, ini, end):
        batch_data = data_tensor[ini: end]
        if Dataset.GPUS:
            batch_data = batch_data.cuda()

        return Variable(batch_data, volatile=self.volatile)

