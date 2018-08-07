#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import figet
from . import utils

log = utils.get_logging()


class ContextEncoder(nn.Module):

    def __init__(self, args):
        self.input_size = args.context_input_size           # 300
        self.rnn_size = args.context_rnn_size               # 200   size of output
        self.num_directions = args.context_num_directions   # 2
        self.num_layers = args.context_num_layers           # 1
        assert self.rnn_size % self.num_directions == 0
        self.hidden_size = self.rnn_size // self.num_directions
        super(ContextEncoder, self).__init__()
        self.rnn = nn.LSTM(self.input_size, self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=args.dropout,
                           bidirectional=(self.num_directions == 2))

    def forward(self, input, word_lut, hidden=None):
        indices = None
        if isinstance(input, tuple):        # yo creo que esto no pasa nunca...
            input, lengths, indices = input

        emb = word_lut(input)   # seq_len x batch x emb
        emb = emb.transpose(0, 1)

        if indices is not None:
            emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(emb, hidden)
        if indices is not None:
            outputs = unpack(outputs)[0]
            outputs = outputs[:,indices, :]
        return outputs, hidden_t


class Attention(nn.Module):

    def __init__(self, args):
        self.args = args
        self.rnn_size = args.context_rnn_size   # 200
        self.attn_size = args.attn_size         # 100
        super(Attention, self).__init__()
        self.linear_in = nn.Linear(args.context_input_size, args.context_rnn_size) # 300, 200
        self.sm = nn.Softmax(dim=0)
        self.tanh = nn.Tanh()

    def forward(self, mention, context):
        context = context.transpose(0, 1).contiguous()
        targetT = self.linear_in(mention).unsqueeze(2)   # batch x attn_size x 1
        attn = torch.bmm(context, targetT).squeeze(2)
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1)) # batch x 1 x seq_len*2
        weighted_context_vec = torch.bmm(attn3, context).squeeze(1)

        if self.args.single_context == 1:
            context_output = weighted_context_vec
        else:
            context_output = self.tanh(weighted_context_vec)

        return context_output, attn


class Classifier(nn.Module):

    def __init__(self, args):
        self.args = args
        self.type_dims = args.type_dims
        self.input_size = args.context_rnn_size + args.context_input_size   # 200 + 300

        super(Classifier, self).__init__()
        self.W = nn.Linear(self.input_size, self.type_dims, bias=args.bias==1)
        # self.sg = nn.Sigmoid()
        self.loss_func = nn.MSELoss()

    def forward(self, input, type_vec=None, type_lut=None):
        logit = self.W(input)   # logit: batch x type_dims
        # distribution = self.sg(logit)
        distribution = logit
        loss = None
        if type_vec is not None:
            type_embeds = type_lut(type_vec)    # batch x type_dims

            loss = self.loss_func(logit, type_embeds)   # should be batch_size x whatever

        return loss, distribution


class MentionEncoder(nn.Module):

    def __init__(self, args):
        super(MentionEncoder, self).__init__()
        self.dropout = nn.Dropout(args.dropout) if args.dropout else None

    def forward(self, input, word_lut):
        if self.dropout:
            return self.dropout(input)
        return input


class Model(nn.Module):

    def __init__(self, args, vocabs):
        self.args = args
        super(Model, self).__init__()
        self.word_lut = nn.Embedding(
            vocabs[figet.Constants.TOKEN_VOCAB].size_of_word2vecs(),
            args.context_input_size,                # context_input_size = 300 (embed dim)
            padding_idx=figet.Constants.PAD
        )

        self.type_lut = nn.Embedding(
            vocabs[figet.Constants.TYPE_VOCAB].size(),
            args.type_dims                         # type_dims = 300
        )

        if args.dropout:
            self.dropout = nn.Dropout(args.dropout)
        else:
            self.dropout = None
        self.mention_encoder = MentionEncoder(args)
        self.prev_context_encoder = ContextEncoder(args)
        self.next_context_encoder = ContextEncoder(args)
        self.attention = Attention(args)
        self.classifier = Classifier(args)

    def init_params(self, word2vec, type2vec):
        self.word_lut.weight.data.copy_(word2vec)
        self.word_lut.weight.requires_grad = False      # by changing this, the weights of the embeddings get updated
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = False

    def forward(self, input):
        mention = input[0]
        prev_context = input[1]
        next_context = input[2]
        type_vec = input[3]

        attn = None
        mention_vec = self.mention_encoder(mention, self.word_lut)
        context_vec, attn = self.encode_context(prev_context, next_context, mention_vec)
        vecs = [mention_vec, context_vec]

        input_vec = torch.cat(vecs, dim=1)
        loss, distribution = self.classifier(input_vec, type_vec, self.type_lut)
        return loss, distribution, attn

    def encode_context(self, prev_context, next_context, mention_vec):
        return self.draw_attention(prev_context, next_context, mention_vec)

    def draw_attention(self, prev_context_vec, next_context_vec, mention_vec):
        if self.args.single_context == 1:
            context_vec, _ = self.prev_context_encoder(prev_context_vec, self.word_lut)
        else:
            prev_context_vec, _ = self.prev_context_encoder(prev_context_vec, self.word_lut)
            next_context_vec, _ = self.next_context_encoder(next_context_vec, self.word_lut)
            context_vec = torch.cat((prev_context_vec, next_context_vec), dim=0)
        weighted_context_vec, attn = self.attention(mention_vec, context_vec)
        return weighted_context_vec, attn
