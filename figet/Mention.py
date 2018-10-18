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

    def preprocess(self, vocabs, args):
        self.vocabs = vocabs
        self.context_len = args.side_context_length
        self.mention_len = args.mention_length
        self.mention_char_len = args.mention_char_length

        self.types = self.type_idx()        # type index in vocab
        self.mention = self.get_mention_idx()
        self.mention_chars = self.get_mention_chars()
        self.left_context = self.left_context_idx()
        self.right_context = self.right_context_idx()

    def get_mention_idx(self):
        head = self.fields[c.HEAD].split()[:self.mention_len]
        if not head:
            return torch.LongTensor([c.PAD])
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(head, c.UNK_WORD)

    def left_context_idx(self):
        left_context_words = self.fields[c.LEFT_CTX].split()[-self.context_len:]
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(left_context_words, c.UNK_WORD)

    def right_context_idx(self):
        right_context_words = self.fields[c.RIGHT_CTX].split()[:self.context_len]
        return self.vocabs[c.TOKEN_VOCAB].convert_to_idx(right_context_words, c.UNK_WORD)

    def type_idx(self):
        types = []
        for full_type in self.fields[c.TYPE]:
            mention_type = clean_type(full_type)
            types.append(self.vocabs[c.TYPE_VOCAB].lookup(mention_type))
        return torch.LongTensor(types)

    def get_mention_chars(self):
        chars = self.fields[c.HEAD][:self.mention_char_len]
        if not chars:
            return torch.LongTensor([c.PAD])
        return self.vocabs[c.CHAR_VOCAB].convert_to_idx(chars, c.UNK_WORD)

    def type_len(self):
        return len(self.fields[c.TYPE])

    def clear(self):
        del self.fields
        del self.mention
        del self.mention_chars
        del self.left_context
        del self.right_context
        del self.types




