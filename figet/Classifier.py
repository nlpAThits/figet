
import torch
import torch.nn as nn
from figet.utils import expand_tensor
from figet.Constants import TYPE_VOCAB
from figet.hyperbolic import PoincareDistance, polarization_identity
from torch.nn import CosineSimilarity
from figet.utils import get_logging, euclidean_dot_product
from math import log as ln

log = get_logging()


class Classifier(nn.Module):
    def __init__(self, args, vocabs, type2vec):
        hidden_size = args.classif_hidden_size
        self.extra_features = [PoincareDistance.apply, CosineSimilarity(), polarization_identity, euclidean_dot_product]
        self.input_size = args.type_dims * 2 + len(self.extra_features)
        self.type_quantity = len(type2vec)
        self.type_dict = vocabs[TYPE_VOCAB]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(Classifier, self).__init__()
        self.type_lut = nn.Embedding(
            vocabs[TYPE_VOCAB].size(),
            args.type_dims
        )
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = False
        self.W1 = nn.Linear(self.input_size, hidden_size, bias=args.classif_bias == 1)
        self.extra_layers = [nn.Linear(hidden_size, hidden_size, bias=args.classif_bias == 1).to(self.device) for _ in range(args.classif_hidden_layers)]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.classif_dropout)
        self.W2 = nn.Linear(hidden_size, 1, bias=args.classif_bias == 1)
        self.sg = nn.Sigmoid()

        self.loss_func = nn.BCEWithLogitsLoss()

        log.debug("Function in classifier: {}".format(self.extra_features))

    def forward(self, type_embeddings, neighbor_indexes, one_hot_neighbor_types=None):
        """
        :param type_embeddings: batch x type_dim
        :param neighbor_indexes: batch x k
        :param one_hot_neighbor_types: batch x k
        :return:
        """
        embeds = self.type_lut(neighbor_indexes)
        neighbor_embeds = embeds.view(embeds.size(0) * embeds.size(1), -1)                  # (batch * k) x type_dim
        expanded_predictions = expand_tensor(type_embeddings, neighbor_indexes.size(1))     # (batch * k) x type_dim

        extra_features = self.get_extra_features(expanded_predictions, neighbor_embeds)
        # popularity_feature = popularity(neighbor_indexes, self.type_dict)

        input = torch.cat((expanded_predictions, neighbor_embeds, extra_features), dim=1).to(self.device)

        hidden_state = self.dropout(self.relu(self.W1(input)))
        for layer in self.extra_layers:
            hidden_state = self.dropout(self.relu(layer(hidden_state)))

        layer_two = self.W2(hidden_state)
        distribution = self.sg(layer_two).view(type_embeddings.size(0), -1)

        one_hot_reshaped = one_hot_neighbor_types.view(one_hot_neighbor_types.size(0) * one_hot_neighbor_types.size(1), -1)

        loss = self.loss_func(layer_two, one_hot_reshaped) if one_hot_neighbor_types is not None else None
        return distribution, loss

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
