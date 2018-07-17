#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import math
import torch
from torch.autograd import Variable

import figet

log = figet.utils.get_logging()


class Dataset(object):

    def __init__(self, data, batch_size, args, volatile=False):
        self.data = data        # list of figet.Mentions
        self.args = args

        self.batch_size = batch_size    # 1000 if train, else len(data)
        self.num_batches = math.ceil(len(self.data) / batch_size)
        self.volatile = volatile
        self.cached_out = None

    def __len__(self):
        return self.num_batches

    def shuffle(self):
        self.data = [self.data[i] for i in torch.randperm(len(self.data))]

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
        if len(self.args.gpus) > 0:
            out = out.cuda()
            mask = mask.cuda()

        out = Variable(out, volatile=self.volatile)
        if include_lengths:
            return out, out_lengths, mask
        return out, None, mask


    # En lugar de pedir por item, debería tener un iterador, y que en cada paso vaya devolviendo un "slice" distinto
    def __getitem__(self, index):
        """
        :param index:
        :return: Matrices of different parts (head string, context) of every instance
        """
        index = int(index % self.num_batches)
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)    # WTF, this is obvious
        batch_data = self.data[index*self.batch_size:(index+1)*self.batch_size]

        # document
        doc_batch = None
        # if self.args.use_doc == 1:
        #     doc_batch = self._batchify_doc([d.doc_vec for d in batch_data])

        # mention
        mention_batch = self._batchify([d.mention for d in batch_data])

        # type
        type_batch = self._batchify([d.types for d in batch_data])

        # context
        if self.args.single_context == 1:       # 0
            context_batch, context_length, mask = self._batchify(
                [d.context for d in batch_data], include_lengths=True)
            # context_batch = self._sort(context_batch, context_length)
            return (
                mention_batch[0],
                (context_batch, mask),
                (None, None),
                type_batch[0], None,
                doc_batch,
                batch_data
            )

        prev_context_batch, prev_context_length, prev_mask = self._batchify(
            [d.prev_context for d in batch_data], self.args.context_length, include_lengths=True)
        next_context_batch, next_context_length, next_mask = self._batchify(
            [d.next_context for d in batch_data], self.args.context_length, include_lengths=True, reverse=True)
        # prev_context_batch = self._sort(prev_context_batch, prev_context_length)
        # next_context_batch = self._sort(next_context_batch, next_context_length)

        return (
            mention_batch[0],
            (prev_context_batch, prev_mask),
            (next_context_batch, next_mask),
            type_batch[0], None,    # type_batch[0], feature_batch[0],
            doc_batch,
            batch_data
        )

    def _batchify_paragraph(self, data):
        """
        NEVER USED
        """
        data = torch.stack(data).float().contiguous()
        if len(self.args.gpus) > 0:
            data = data.cuda()
        data = Variable(data, volatile=self.volatile)
        return data

    def _sort(self, batch_data, lengths):
        """
        NEVER USED
        """
        lengths = torch.LongTensor(lengths)
        lengths, indices = torch.sort(lengths, dim=0, descending=True)
        lengths = lengths.numpy()
        if len(self.args.gpus) > 0:
            indices = indices.cuda()
        batch_data = batch_data[indices, :]
        _, indices = torch.sort(indices, dim=0)
        return batch_data, lengths, indices