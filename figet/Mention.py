#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import torch
import figet
from Constants import TOKEN_VOCAB, TYPE_VOCAB

# fields keys
HEAD = "mid"
RIGHT_CTX = "rCtx"
LEFT_CTX = "lCtx"
TYPE = "type"


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

        self.types = self.fields[TYPE]

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
        words2vecs = [self.word2vec[self.vocabs[TOKEN_VOCAB].lookup(token, figet.Constants.UNK)] for token in self.fields[HEAD]]
        return torch.mean(torch.stack(words2vecs), dim=0).squeeze(0)

    def context_idx(self):
        context = (self.prev_context_words() +
                   [figet.Constants.PAD_WORD] +
                   self.next_context_words())
        return self.vocabs[TOKEN_VOCAB].convert_to_idx(context, figet.Constants.UNK_WORD)

    def prev_context_idx(self):
        prev_context = self.prev_context_words()
        return self.vocabs[TOKEN_VOCAB].convert_to_idx(prev_context, figet.Constants.UNK_WORD)

    def prev_context_words(self):
        return self.fields[LEFT_CTX].split()[-self.context_length:]

    def next_context_idx(self):
        next_context = self.next_context_words()
        return self.vocabs[TOKEN_VOCAB].convert_to_idx(next_context, figet.Constants.UNK_WORD)

    def next_context_words(self):
        return self.fields[RIGHT_CTX].split()[:self.context_length]

    def type_idx(self):
        type_vec = torch.Tensor(self.vocabs[TYPE_VOCAB].size()).zero_()
        # For the way we are modeling the data, every mention has only one type (that will have an impact on the
        # metrics as well. For now I keep the loop, in case we go back to multiple types per Mention
        for type_ in self.types:
            type_idx = self.vocabs[TYPE_VOCAB].lookup(type_)
            type_vec[type_idx] = 1
        return type_vec
