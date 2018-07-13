#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import torch
import figet.Constants as c


class Mention(object):

    def __init__(self, fields, doc_vec=None):
        """
        :param fields: json lines with lCtx, rCtx, mid and type
        :param doc_vec:
        """
        self.fields = fields
        self.doc_vec = doc_vec

    def preprocess(self, vocabs, word2vec, args):
        self.vocabs = vocabs
        self.word2vec = word2vec
        self.context_length = args.context_length   # 10 by default

        self.types = self.fields[c.TYPE]

        self.types = self.type_idx()        # types is a one-hot vector of the types of this mention
        self.mention = self.mention_idx()   # average of embeds forming the mention
        if args.single_context == 1:
            self.context = self.context_idx()
        else:
            self.prev_context = self.prev_context_idx()
            self.next_context = self.next_context_idx()
        # if self.doc_vec is not None:
        #     self.doc_vec = torch.from_numpy(self.doc_vec)
        self.tokens = None

    def mention_idx(self):
        words2vecs = [self.word2vec[self.vocabs[c.TOKEN_VOCAB].lookup(token, c.UNK)] for token in self.fields[c.HEAD]]
        return torch.mean(torch.stack(words2vecs), dim=0).squeeze(0)

    def context_idx(self):
        context = (self.prev_context_words() + [c.PAD_WORD] + self.next_context_words())
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(context, c.UNK_WORD)

    def prev_context_idx(self):
        prev_context = self.prev_context_words()
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(prev_context, c.UNK_WORD)

    def prev_context_words(self):
        return self.fields[c.LEFT_CTX].split()[-self.context_length:]

    def next_context_idx(self):
        next_context = self.next_context_words()
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(next_context, c.UNK_WORD)

    def next_context_words(self):
        return self.fields[c.RIGHT_CTX].split()[:self.context_length]

    def type_idx(self):
        type_vec = torch.Tensor(self.vocabs[c.TYPE_VOCAB].size()).zero_()
        for type_ in self.types:
            type_idx = self.vocabs[c.TYPE_VOCAB].lookup(type_)
            type_vec[type_idx] = 1
        return type_vec

    def __str__(self):
        type_labels = []
        for idx in self.types:
            if idx != 0:
                type_labels.append(self.vocabs[c.TYPE_VOCAB].get_label(idx))
        return "Mention: {}, types: {}".format(self.fields[c.HEAD], type_labels)