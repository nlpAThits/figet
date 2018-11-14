
import torch
import torch.nn as nn
from figet.utils import expand_tensor
from figet.Constants import TYPE_VOCAB
from figet.hyperbolic import PoincareDistance, hyperbolic_norm
from torch.nn import CosineSimilarity
from figet.utils import get_logging, euclidean_dot_product
from math import log as ln

log = get_logging()


class Classifier(nn.Module):
    def __init__(self, args, vocabs, type2vec):
        self.type_quantity = len(type2vec)
        self.extra_features = [PoincareDistance.apply, CosineSimilarity(), euclidean_dot_product, nn.PairwiseDistance(),
                               neighbors_norm]
        feature_repre_size = args.context_rnn_size * 2 + args.emb_size + args.char_emb_size   # 200 * 2 + 300 + 50
        self.input_size = feature_repre_size + args.type_dims + (args.type_dims + len(self.extra_features)) * args.neighbors + + self.type_quantity
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

        log.debug("Function in classifier: {}".format(self.extra_features))

    def forward(self, predicted_embeds, neighbor_indexes, feature_repre, one_hot_neighbor_types=None):
        """
        :param predicted_embeds: batch x type_dim
        :param neighbor_indexes: batch x k
        :param feature_repre: batch x feature_size
        :param one_hot_neighbor_types: batch x k
        :return:
        """
        embeds = self.type_lut(neighbor_indexes)
        neighbor_embeds = embeds.view(embeds.size(0) * embeds.size(1), -1)                  # (batch * k) x type_dim
        expanded_predictions = expand_tensor(predicted_embeds, neighbor_indexes.size(1))     # (batch * k) x type_dim

        extra_features = self.get_extra_features(expanded_predictions, neighbor_embeds)     # (batch * k) x len(extra_features)

        neighbors_and_features = torch.cat((neighbor_embeds, extra_features), dim=1).to(self.device)
        neighbor_repre = neighbors_and_features.view(len(predicted_embeds), -1)      # batch x (type_dim + extra_feat) * k

        one_hot_neighbors_all_types = torch.zeros(predicted_embeds.size(0), self.type_quantity).to(self.device)
        one_hot_neighbors_all_types.scatter_(1, neighbor_indexes, 1.0)

        input = torch.cat((feature_repre, predicted_embeds, neighbor_repre, one_hot_neighbors_all_types), dim=1).to(self.device)

        logit = self.W(input)                                      # batch x type_quantity
        distribution = self.sg(logit)

        # keep only the neighboring types
        logit_neigh = logit.gather(1, neighbor_indexes)
        distribution_neigh = distribution.gather(1, neighbor_indexes)

        loss = self.loss_func(logit_neigh, one_hot_neighbor_types) if one_hot_neighbor_types is not None else None
        return distribution_neigh, loss

    def get_extra_features(self, expanded_predictions, true_types):
        result = self.extra_features[0](expanded_predictions, true_types).unsqueeze(1)
        for f in self.extra_features[1:]:
            partial = f(expanded_predictions, true_types)
            result = torch.cat((result, partial.unsqueeze(1)), dim=1)

        return result


def popularity(neighbor_indexes, type_dict):
    indexes = neighbor_indexes.view(neighbor_indexes.size(0) * neighbor_indexes.size(1), -1)
    type_popularity = torch.Tensor(indexes.size()).to('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(len(indexes)):
        idx = indexes[i].item()
        freq = type_dict.frequencies[idx]
        type_popularity[i] = ln(1 + freq)
    return type_popularity


def neighbors_norm(predictions, neighbors):
    return hyperbolic_norm(neighbors)
