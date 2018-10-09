#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from figet import Constants
from figet.hyperbolic import PoincareDistance
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
        self.W = nn.Linear(self.input_size, args.type_dims, bias=args.proj_bias == 1)
        self.activation_function = extra_args["activation_function"] if "activation_function" in extra_args else None

    def forward(self, input):
        output = self.W(input)  # batch x type_dims
        if self.activation_function:
            output = self.activation_function(output)
        return output


class Model(nn.Module):

    def __init__(self, args, vocabs, negative_samples, extra_args):
        self.args = args
        self.negative_samples = negative_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.distance_function = PoincareDistance.apply

    def init_params(self, word2vec, type2vec):
        self.word_lut.weight.data.copy_(word2vec)
        self.word_lut.weight.requires_grad = False      # by changing this, the weights of the embeddings get updated
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = False

    def forward(self, input, epoch=None):
        mention, prev_context, next_context = input[0], input[1], input[2]
        type_indexes = input[3]

        mention_vec = self.mention_encoder(mention, self.word_lut)
        context_vec, attn = self.encode_context(prev_context, next_context, mention_vec)

        input_vec = torch.cat([mention_vec, context_vec], dim=1)
        predicted_emb = self.projector(input_vec)

        normalized_emb = normalize(predicted_emb)

        loss, avg_neg_dist, dist_to_pos, dist_to_neg, euclid_dist = 0, 0, 0, 0, 0
        if type_indexes is not None:
            loss, avg_neg_dist, dist_to_pos, dist_to_neg, euclid_dist = self.calculate_loss(normalized_emb, type_indexes, epoch)

        return loss, normalized_emb, attn, avg_neg_dist, dist_to_pos, dist_to_neg, euclid_dist

    def calculate_loss(self, predicted_embeds, type_indexes, epoch=None):
        type_len = type_indexes.size(1)             # It is the same for the whole batch
        type_embeds = self.type_lut(type_indexes)   # batch x type_dims
        true_type_embeds = type_embeds.view(type_embeds.size(0) * type_embeds.size(1), -1)

        expanded_predicted = utils.expand_tensor(predicted_embeds, type_len)

        distances_to_pos = self.distance_function(expanded_predicted, true_type_embeds)
        distances_to_neg = self.get_negative_sample_distances(predicted_embeds, type_indexes, epoch)

        distances = torch.cat((distances_to_pos, distances_to_neg))
        sq_distances = distances ** 2

        y = torch.ones(len(sq_distances)).to(self.device)
        y[len(distances_to_pos):] = -1

        avg_neg_distance = self.get_average_negative_distance(type_indexes, epoch)
        loss_func = nn.HingeEmbeddingLoss(margin=(avg_neg_distance * 0.6)**2)

        # stats
        euclid_func = nn.PairwiseDistance()
        euclid_dist = euclid_func(expanded_predicted, true_type_embeds)

        return loss_func(sq_distances, y), avg_neg_distance, distances_to_pos, distances_to_neg, euclid_dist

    def get_negative_sample_distances(self, predicted_embeds, type_vec, epoch=None):
        neg_sample_indexes = []
        for i in range(len(predicted_embeds)):
            type_index = type_vec[i][-1].item()     # the last one because tends to be the more specific one
            neg_indexes = self.negative_samples.get_indexes(type_index, self.args.negative_samples, epoch, self.args.epochs)
            neg_sample_indexes.extend(neg_indexes)

        neg_type_vecs = self.type_lut(torch.LongTensor(neg_sample_indexes).to(self.device))
        expanded_predicted_embeds = utils.expand_tensor(predicted_embeds, self.args.negative_samples)

        return self.distance_function(expanded_predicted_embeds, neg_type_vecs)

    def get_average_negative_distance(self, type_vec, epoch):
        distances = []
        for i in range(len(type_vec)):
            type_index = type_vec[i][0].item()
            distances.extend(self.negative_samples.get_distances(type_index, self.args.negative_samples, epoch, self.args.epochs))

        return (sum(distances) / len(distances)).item()

    def encode_context(self, prev_context, next_context, mention_vec):
        if self.args.single_context == 1:
            context_vec, _ = self.prev_context_encoder(prev_context, self.word_lut)
        else:
            prev_context_vec, _ = self.prev_context_encoder(prev_context, self.word_lut)
            next_context_vec, _ = self.next_context_encoder(next_context, self.word_lut)
            context_vec = torch.cat((prev_context_vec, next_context_vec), dim=0)
        weighted_context_vec, attn = self.attention(mention_vec, context_vec)
        return weighted_context_vec, attn


def normalize(predicted_emb):
    norms = torch.sqrt(torch.sum(predicted_emb * predicted_emb, dim=-1))
    indexes = norms >= 1
    norms = norms * (1 + Constants.EPS)
    inverses = 1.0 / norms
    inverses = inverses * indexes.float()
    complement = indexes == 0
    inverses = inverses + complement.float()
    stacked_inverses = torch.stack([inverses] * predicted_emb.size(1), 1)
    return predicted_emb * stacked_inverses

