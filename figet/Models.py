#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from figet import Constants
from figet.hyperbolic import PoincareDistance, normalize
from . import utils
from figet.model_utils import CharEncoder, SelfAttentiveSum, sort_batch_by_length
from math import pi

log = utils.get_logging()


class MentionEncoder(nn.Module):

    def __init__(self, char_vocab, args):
        super(MentionEncoder, self).__init__()
        self.char_encoder = CharEncoder(char_vocab, args)
        self.attentive_weighted_average = SelfAttentiveSum(args.emb_size, 1)
        self.dropout = nn.Dropout(args.mention_dropout)

    def forward(self, mentions, mention_chars, word_lut):
        mention_embeds = word_lut(mentions)             # batch x mention_length x emb_size

        weighted_avg_mentions, _ = self.attentive_weighted_average(mention_embeds)
        char_embed = self.char_encoder(mention_chars)
        output = torch.cat((weighted_avg_mentions, char_embed), 1)
        return self.dropout(output)


class ContextEncoder(nn.Module):

    def __init__(self, args):
        self.emb_size = args.emb_size                       # 300
        self.pos_emb_size = args.positional_emb_size        # 50
        self.rnn_size = args.context_rnn_size               # 200   size of output
        self.hidden_attention_size = 100
        super(ContextEncoder, self).__init__()
        self.rnn = nn.LSTM(self.emb_size + self.pos_emb_size, self.rnn_size, bidirectional=True, batch_first=True)
        self.pos_linear = nn.Linear(1, self.pos_emb_size)
        self.attention = SelfAttentiveSum(self.rnn_size * 2, self.hidden_attention_size) # x2 because of bidirectional

    def forward(self, contexts, positions, context_len, word_lut, hidden=None):
        """
        :param contexts: batch x max_seq_len
        :param positions: batch x max_seq_len
        :param context_len: batch x 1
        """
        positional_embeds = self.get_positional_embeddings(positions)   # batch x max_seq_len x pos_emb_size
        ctx_word_embeds = word_lut(contexts)                            # batch x max_seq_len x emb_size
        ctx_embeds = torch.cat((ctx_word_embeds, positional_embeds), 2)

        rnn_output = self.sorted_rnn(ctx_embeds, context_len)

        return self.attention(rnn_output)

    def get_positional_embeddings(self, positions):
        """ :param positions: batch x max_seq_len"""
        pos_embeds = self.pos_linear(positions.view(-1, 1))                     # batch * max_seq_len x pos_emb_size
        return pos_embeds.view(positions.size(0), positions.size(1), -1)        # batch x max_seq_len x pos_emb_size

    def sorted_rnn(self, ctx_embeds, context_len):
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(ctx_embeds, context_len)
        packed_sequence_input = pack(sorted_inputs, sorted_sequence_lengths, batch_first=True)
        packed_sequence_output, _ = self.rnn(packed_sequence_input, None)
        unpacked_sequence_tensor, _ = unpack(packed_sequence_output, batch_first=True)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)


class Attention(nn.Module):
    """
    DEPRECATED for now...
    """

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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = args.context_rnn_size * 2 + args.emb_size + args.char_emb_size   # 200 * 2 + 300 + 50
        self.hidden_size = args.proj_hidden_size
        super(Projector, self).__init__()


        # ACA usan una sola capa linear con una ReLU al final... No me gusta que sea ReLU porq hace que todos los valores den positivos
        # Una TanH sería una con imagen entre [-1;1], pero se lo puede dejar como hyperparametro


        self.W_in = nn.Linear(self.input_size, self.hidden_size, bias=args.proj_bias == 1)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=args.proj_bias == 1)
                                            for _ in range(args.proj_hidden_layers)])
        self.W_out = nn.Linear(self.hidden_size, args.type_dims, bias=args.proj_bias == 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.proj_dropout)

    def forward(self, input):
        hidden_state = self.dropout(self.relu(self.W_in(input)))
        for layer in self.hidden_layers:
            hidden_state = self.dropout(self.relu(layer(hidden_state)))

        return self.W_out(hidden_state)  # batch x type_dims


class Model(nn.Module):

    def __init__(self, args, vocabs, negative_samples, extra_args):
        self.args = args
        self.negative_samples = negative_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(Model, self).__init__()
        self.word_lut = nn.Embedding(vocabs[Constants.TOKEN_VOCAB].size_of_word2vecs(), args.emb_size,
                                     padding_idx=Constants.PAD)
        self.type_lut = nn.Embedding(vocabs[Constants.TYPE_VOCAB].size(), args.type_dims)
        self.linear_transf = nn.Linear(1, 1, True)

        self.mention_encoder = MentionEncoder(vocabs[Constants.CHAR_VOCAB], args)
        self.context_encoder = ContextEncoder(args)
        self.projector = Projector(args, extra_args)
        self.distance_function = PoincareDistance.apply
        self.loss_function = nn.HingeEmbeddingLoss(margin=self.args.hinge_margin, reduction='sum')

    def init_params(self, word2vec, type2vec):
        self.word_lut.weight.data.copy_(word2vec)
        self.word_lut.weight.requires_grad = False      # by changing this, the weights of the embeddings get updated
        nn.init.normal_(self.type_lut.weight.data, mean=0, std=0.01)
        self.type_lut.weight.requires_grad = True

    def forward(self, input, epoch=None):
        contexts, positions, context_len = input[0], input[1], input[2]
        mentions, mention_chars = input[3], input[4]
        type_indexes = input[5]

        mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
        context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

        input_vec = torch.cat((mention_vec, context_vec), dim=1)

        predicted_emb = self.projector(input_vec)
        normalized_emb = normalize(predicted_emb)

        loss, avg_angle, distance_to_pos, euclid_dist = 0, 0, 0, 0

        if type_indexes is not None:
            distance_to_pos, distance_to_neg, avg_angle, euclid_dist = self.get_hyperbolic_distances(normalized_emb, type_indexes)
            pos_transformed = self.linear_transf(distance_to_pos.unsqueeze(1))
            neg_transformed = self.linear_transf(distance_to_neg.unsqueeze(1))

            loss = self.calculate_loss(pos_transformed, neg_transformed)

        return loss, normalized_emb, input_vec, attn, avg_angle, distance_to_pos, euclid_dist

    def get_hyperbolic_distances(self, predicted_embeds, type_indexes):
        """
        :param predicted_embeds: batch x type_dims
        :param type_indexes: batch x type_len (quantity of true types in this batch)
        :return:
        """
        type_len = type_indexes.size(1)  # Is the same for the whole batch
        type_embeds = self.type_lut(type_indexes)
        positive_type_embeds = type_embeds.view(type_embeds.size(0) * type_embeds.size(1), -1)  # batch * type_len x type_dims
        # expanded_pos_type_embeds = utils.expand_tensor(positive_type_embeds, self.args.negative_samples)  # batch * type_len * neg_sample x type_dims

        negative_type_embeds = self.get_negative_samples(predicted_embeds, type_indexes)  # batch * type_len * neg_sample x type_dim

        expanded_predicted_for_pos = utils.expand_tensor(predicted_embeds, type_len)  # batch * type_len x type_dims
        expanded_predicted_for_neg = utils.expand_tensor(predicted_embeds, type_len * self.args.negative_samples)  # batch * type_len * neg_sample x type_dims

        dist_to_pos = self.distance_function(expanded_predicted_for_pos, positive_type_embeds)
        dist_to_neg = self.distance_function(expanded_predicted_for_neg, negative_type_embeds)
        dist_to_pos_expanded = utils.expand_tensor(dist_to_pos.unsqueeze(1), self.args.negative_samples).squeeze()

        # stats
        cos_sim = nn.CosineSimilarity()
        avg_angle = torch.acos(cos_sim(expanded_predicted_for_pos, positive_type_embeds)) * 180 / pi
        euclidean_dist_func = nn.PairwiseDistance()
        euclid_dist = euclidean_dist_func(expanded_predicted_for_pos, positive_type_embeds)

        return dist_to_pos_expanded, dist_to_neg, avg_angle, euclid_dist

    def get_negative_samples(self, predicted_embeds, true_type_indexes, epoch=None):
        """
        :param predicted_embeds: batch x type_dim
        :param true_type_indexes: batch x type_len
        :param epoch:
        :return:
        """
        neg_sample_indexes = []
        for i in range(len(predicted_embeds)):
            for positive_type_index in true_type_indexes[i]:
                neg_indexes = self.negative_samples.get_indexes(positive_type_index.item(), self.args.negative_samples, epoch, self.args.epochs)
                neg_sample_indexes.extend(neg_indexes)

        return self.type_lut(torch.LongTensor(neg_sample_indexes).to(self.device))  # batch * type_len * neg_sample x type_dim

    def get_average_negative_distance(self, type_vec, epoch):
        """DEPRECATED"""
        distances = []
        for i in range(len(type_vec)):
            type_index = type_vec[i][0].item()
            distances.extend(self.negative_samples.get_distances(type_index, self.args.negative_samples, epoch, self.args.epochs))

        return (sum(distances) / len(distances)).item()

    def calculate_loss(self, pos_samples, neg_samples):
        value = pos_samples + self.args.hinge_margin - neg_samples
        y = torch.ones(len(pos_samples)).to(self.device)
        return self.loss_function(value, y)

    def normalize_type_embeddings(self):
        self.type_lut.weight.data.copy_(normalize(self.type_lut.weight.data))

    def get_type_embeds(self):
        return self.type_lut.weight.data

    def get_linear_transf_weights(self):
        return self.linear_transf.weight.data.item(), self.linear_transf.bias.data.item()
