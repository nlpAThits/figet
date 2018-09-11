#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import torch
import figet.Constants as c
from figet.utils import clean_type


class Mention(object):

    def __init__(self, fields):
        """
        :param fields: json lines with lCtx, rCtx, mid and type
        """
        self.fields = fields

    def preprocess(self, vocabs, word2vec, args):
        self.vocabs = vocabs
        self.word2vec = word2vec
        self.context_length = args.context_length   # 10 by default

        self.types = self.type_idx()        # type index in vocab
        self.mention_idx = self.get_mention_idx()
        self.mention = self.mention_avg()   # average of embeds forming the mention
        if args.single_context == 1:
            self.context = self.context_idx()
        else:
            self.prev_context = self.prev_context_idx()
            self.next_context = self.next_context_idx()
        self.tokens = None

    def mention_avg(self):
        words2vecs = [self.word2vec[self.vocabs[c.TOKEN_VOCAB].lookup(token, c.PAD)] for token in self.fields[c.HEAD].split()]
        return torch.mean(torch.stack(words2vecs), dim=0).squeeze(0)

    def get_mention_idx(self):
        head = self.fields[c.HEAD].split()[:self.context_length]
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(head, c.PAD_WORD)

    def context_idx(self):
        context = (self.prev_context_words() + [c.PAD_WORD] + self.next_context_words())
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(context, c.PAD_WORD)

    def prev_context_idx(self):
        prev_context = self.prev_context_words()
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(prev_context, c.PAD_WORD)

    def prev_context_words(self):
        return self.fields[c.LEFT_CTX].split()[-self.context_length:]

    def next_context_idx(self):
        next_context = self.next_context_words()
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(next_context, c.PAD_WORD)

    def next_context_words(self):
        return self.fields[c.RIGHT_CTX].split()[:self.context_length]

    def type_idx(self):
        types = []
        for full_type in self.fields[c.TYPE]:
            mention_type = clean_type(full_type)
            types.append(self.vocabs[c.TYPE_VOCAB].lookup(mention_type))
        return torch.LongTensor(types)

    def type_len(self):
        return len(self.fields[c.TYPE])

    def clear(self):
        del self.fields
        del self.mention
        del self.types
        del self.context_length
        try:
            del self.context
        except AttributeError:
            del self.prev_context
            del self.next_context


