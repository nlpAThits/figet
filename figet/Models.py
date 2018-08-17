#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from figet.hyperbolic_parameter import HyperbolicParameter
from figet import Constants
from . import utils

log = utils.get_logging()


class MentionEncoder(nn.Module):

    def __init__(self, args):
        super(MentionEncoder, self).__init__()
        self.dropout = nn.Dropout(args.dropout) if args.dropout else None

    def forward(self, input, word_lut):
        if self.dropout:
            return self.dropout(input)
        return input


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


class Projector(nn.Module):

    def __init__(self, args, extra_args):
        self.args = args
        self.input_size = args.context_rnn_size + args.context_input_size   # 200 + 300
        super(Projector, self).__init__()
        self.W = nn.Linear(self.input_size, args.type_dims, bias=args.bias == 1)
        self.activation_function = extra_args["activation_function"] if "activation_function" in extra_args else None

    def forward(self, input):
        logit = self.W(input)  # logit: batch x type_dims
        if self.activation_function:
            return self.activation_function(logit)
        return logit


class Model(nn.Module):

    def __init__(self, args, vocabs, extra_args):
        self.args = args
        super(Model, self).__init__()
        self.word_lut = nn.Embedding(
            vocabs[Constants.TOKEN_VOCAB].size_of_word2vecs(),
            args.context_input_size,                # context_input_size = 300 (embed dim)
            padding_idx=Constants.PAD
        )

        self.type_lut = nn.Embedding(
            vocabs[Constants.TYPE_VOCAB].size(),
            args.type_dims
        )

        self.mention_encoder = MentionEncoder(args)
        self.prev_context_encoder = ContextEncoder(args)
        self.next_context_encoder = ContextEncoder(args)
        self.attention = Attention(args)
        self.projector = Projector(args, extra_args)

        if extra_args["loss_metric"]:
            self.distance_function = extra_args["loss_metric"]
        else:
            self.distance_function = nn.PairwiseDistance(p=2, eps=np.finfo(float).eps) # euclidean distance
        self.loss_func = nn.HingeEmbeddingLoss()

    def init_params(self, word2vec, type2vec):
        self.word_lut.weight.data.copy_(word2vec)
        self.word_lut.weight.requires_grad = False      # by changing this, the weights of the embeddings get updated
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = False

    def forward(self, input):
        mention, prev_context, next_context = input[0], input[1], input[2]
        type_vec = input[3]

        mention_vec = self.mention_encoder(mention, self.word_lut)
        context_vec, attn = self.encode_context(prev_context, next_context, mention_vec)

        input_vec = torch.cat([mention_vec, context_vec], dim=1)
        predicted_emb = self.projector(input_vec)

        normalized_emb = self.normalize(predicted_emb)

        loss = 0
        if type_vec is not None:
            loss = self.calculate_loss(predicted_emb, type_vec)

        return loss, predicted_emb, attn

    def log_grads(self):
        log.debug("Predicted:")
        log.debug(self.predicted)
        log.debug("Predicted grads:")
        log.debug(self.predicted.grad)

        # log.debug("Normalized:")
        # log.debug(self.normalized)
        # log.debug("Normalized grads:")
        # log.debug(self.normalized.grad)

    def normalize(self, predicted_emb):
        norms = torch.sqrt(torch.sum(predicted_emb * predicted_emb, dim=-1))
        indexes = norms >= 1
        norms *= (1 + Constants.EPS)
        inverses = 1.0 / norms
        inverses *= indexes.float()
        complement = indexes == 0
        inverses += complement.float()
        stacked_inverses = torch.stack([inverses] * predicted_emb.size()[1], 1)
        return predicted_emb * stacked_inverses

    def calculate_loss(self, predicted_embeds, type_vec):
        true_type_embeds = self.type_lut(type_vec)  # batch x type_dims

        distances = self.distance_function(predicted_embeds, true_type_embeds)

        sq_distances = distances ** 2

        y = torch.ones(len(sq_distances))
        if torch.cuda.is_available():
            y = y.cuda()

        return self.loss_func(sq_distances, y)  # batch_size x type_dims

    def encode_context(self, prev_context, next_context, mention_vec):
        if self.args.single_context == 1:
            context_vec, _ = self.prev_context_encoder(prev_context, self.word_lut)
        else:
            prev_context_vec, _ = self.prev_context_encoder(prev_context, self.word_lut)
            next_context_vec, _ = self.next_context_encoder(next_context, self.word_lut)
            context_vec = torch.cat((prev_context_vec, next_context_vec), dim=0)
        weighted_context_vec, attn = self.attention(mention_vec, context_vec)
        return weighted_context_vec, attn
