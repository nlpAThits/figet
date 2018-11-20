
import torch
import torch.nn as nn
from figet.Constants import TYPE_VOCAB


class Classifier(nn.Module):
    def __init__(self, args, vocabs, type2vec):
        self.type_quantity = len(type2vec)
        feature_repre_size = args.context_rnn_size * 2 + args.emb_size + args.char_emb_size   # 200 * 2 + 300 + 50
        self.input_size = args.type_dims + feature_repre_size + self.type_quantity
        self.type_dict = vocabs[TYPE_VOCAB]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(Classifier, self).__init__()
        self.type_lut = nn.Embedding(
            vocabs[TYPE_VOCAB].size(),
            args.type_dims
        )
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = False

        self.W = nn.Linear(self.input_size, self.type_quantity, bias=args.classif_bias == 1)
        self.sg = nn.Sigmoid()

        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, predicted_embeds, neighbor_indexes, feature_repre, one_hot_neighbor_types=None):
        """
        :param predicted_embeds: batch x type_dim
        :param neighbor_indexes: batch x k
        :param feature_repre: batch x feature_size
        :param one_hot_neighbor_types: batch x k
        :return:
        """
        one_hot_neighbors_all_types = torch.zeros(predicted_embeds.size(0), self.type_quantity).to(self.device)
        one_hot_neighbors_all_types.scatter_(1, neighbor_indexes, 1.0)

        input = torch.cat((predicted_embeds, feature_repre, one_hot_neighbors_all_types), dim=1).to(self.device)

        logit = self.W(input)                                      # batch x type_quantity
        distribution = self.sg(logit)

        # keep only the neighboring types
        logit_neigh = logit.gather(1, neighbor_indexes)
        distribution_neigh = distribution.gather(1, neighbor_indexes)

        loss = self.loss_func(logit_neigh, one_hot_neighbor_types) if one_hot_neighbor_types is not None else None
        return distribution_neigh, loss