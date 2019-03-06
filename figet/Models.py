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
        self.pos_linear = nn.Linear(1, self.pos_emb_size)
        self.context_dropout = nn.Dropout(args.context_dropout)
        self.rnn = nn.LSTM(self.emb_size + self.pos_emb_size, self.rnn_size, bidirectional=True, batch_first=True)
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

        ctx_embeds = self.context_dropout(ctx_embeds)

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

    def __init__(self, args, extra_args, input_size):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_size = args.proj_hidden_size
        super(Projector, self).__init__()
        self.W_in = nn.Linear(input_size, self.hidden_size, bias=args.proj_bias == 1)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=args.proj_bias == 1)
                                            for _ in range(args.proj_hidden_layers)])
        self.W_out = nn.Linear(self.hidden_size, args.type_dims, bias=args.proj_bias == 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.proj_dropout)
        for layer in [self.W_in, self.W_out] + [l for l in self.hidden_layers]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, input):
        hidden_state = self.dropout(self.relu(self.W_in(input)))
        for layer in self.hidden_layers:
            hidden_state = self.dropout(self.relu(layer(hidden_state)))

        output = self.W_out(hidden_state)  # batch x type_dims

        return normalize(output)


class Model(nn.Module):

    def __init__(self, args, vocabs, negative_samples, extra_args):
        self.args = args
        self.negative_samples = negative_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        type_vocab = vocabs[Constants.TYPE_VOCAB]
        self.coarse_ids = type_vocab.get_coarse_ids()
        self.fine_ids = type_vocab.get_fine_ids()
        self.ultrafine_ids = type_vocab.get_ultrafine_ids()
        self.ids = [self.coarse_ids, self.fine_ids, self.ultrafine_ids]

        super(Model, self).__init__()
        self.word_lut = nn.Embedding(vocabs[Constants.TOKEN_VOCAB].size_of_word2vecs(), args.emb_size, padding_idx=Constants.PAD)
        self.type_lut = nn.Embedding(vocabs[Constants.TYPE_VOCAB].size(), args.type_dims)

        self.mention_encoder = MentionEncoder(vocabs[Constants.CHAR_VOCAB], args)
        self.context_encoder = ContextEncoder(args)
        self.feature_len = args.context_rnn_size * 2 + args.emb_size + args.char_emb_size   # 200 * 2 + 300 + 50

        self.coarse_projector = Projector(args, extra_args, self.feature_len)
        self.fine_projector = Projector(args, extra_args, self.feature_len + args.type_dims)
        self.ultrafine_projector = Projector(args, extra_args, self.feature_len + args.type_dims)

        self.distance_function = PoincareDistance.apply
        self.hinge_loss_func = nn.HingeEmbeddingLoss()

    def init_params(self, word2vec, type2vec):
        self.word_lut.weight.data.copy_(word2vec)
        self.word_lut.weight.requires_grad = False      # by changing this, the weights of the embeddings get updated
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = True

    def forward(self, input, epoch=None):
        contexts, positions, context_len = input[0], input[1], input[2]
        mentions, mention_chars = input[3], input[4]
        type_indexes = input[5]

        mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
        context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

        input_vec = torch.cat((mention_vec, context_vec), dim=1)

        coarse_embed = self.coarse_projector(input_vec)

        fine_input = torch.cat((input_vec, coarse_embed), dim=1)
        fine_embed = self.fine_projector(fine_input)

        ultrafine_input = torch.cat((input_vec, fine_embed), dim=1)
        ultrafine_embed = self.ultrafine_projector(ultrafine_input)
        pred_embeddings = [coarse_embed, fine_embed, ultrafine_embed]

        final_loss = 0
        loss, avg_angle, dist_to_pos, euclid_dist = [], [], [], []
        if type_indexes is not None:
            for predictions, ids in zip(pred_embeddings, self.ids):
                loss_i, avg_angle_i, dist_to_pos_i, euclid_dist_i = self.calculate_loss(predictions, type_indexes, ids, epoch)
                loss.append(loss_i)
                avg_angle.append(avg_angle_i)
                dist_to_pos.append(dist_to_pos_i)
                euclid_dist.append(euclid_dist_i)

            final_loss = sum(loss)

        return final_loss, pred_embeddings, input_vec, attn, avg_angle, dist_to_pos, euclid_dist

    def calculate_loss(self, predicted_embeds, type_indexes, granularity_ids, epoch=None):
        types_by_instance = self.get_types_by_instance(type_indexes, granularity_ids)

        type_lut_ids = [idx for row in types_by_instance for idx in row]
        index_on_prediction = self.get_index_on_prediction(types_by_instance)

        if len(type_lut_ids) == 0:
            return torch.zeros(1, requires_grad=True).to(self.device), torch.zeros(1).to(self.device), \
                   torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)

        true_type_embeds = self.type_lut(torch.LongTensor(type_lut_ids).to(self.device))     # len_type_lut_ids x type_dims

        expanded_predicted = predicted_embeds[index_on_prediction]

        distances_to_pos = self.distance_function(expanded_predicted, true_type_embeds)
        sq_distances = distances_to_pos ** 2

        cos_sim_func = nn.CosineSimilarity()
        cosine_similarity = cos_sim_func(expanded_predicted, true_type_embeds)
        cosine_distance = 1 - cosine_similarity

        total_distance = self.args.hyperdist_factor * sq_distances + self.args.cosine_factor * cosine_distance
        y = torch.ones(len(expanded_predicted)).to(self.device)
        loss = self.hinge_loss_func(total_distance, y)

        # stats
        avg_angle = torch.acos(torch.clamp(cosine_similarity.detach(), min=-1, max=1)) * 180 / pi
        euclidean_dist_func = nn.PairwiseDistance()
        euclid_dist = euclidean_dist_func(expanded_predicted.detach(), true_type_embeds.detach())

        return loss, avg_angle, distances_to_pos.detach(), euclid_dist

    def get_types_by_instance(self, type_indexes, gran_ids):
        result = []
        for row in type_indexes:
            row_result = [idx for idx in row.tolist() if idx in gran_ids]
            result.append(row_result)
        return result

    def get_index_on_prediction(self, types_by_instance):
        indexes = []
        for index, instance in enumerate(types_by_instance):
            for i in range(len(instance)):
                indexes.append(index)
        return indexes

    def normalize_type_embeddings(self):
        self.type_lut.weight.data.copy_(normalize(self.type_lut.weight.data))

    def get_type_embeds(self):
        return self.type_lut.weight.data
