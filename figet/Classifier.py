
import torch
import torch.nn as nn
from figet.utils import expand_tensor
from figet.Constants import TYPE_VOCAB
from figet.hyperbolic import PoincareDistance, polarization_identity
from torch.nn import CosineSimilarity
from figet.utils import get_logging

log = get_logging()


class Classifier(nn.Module):
    def __init__(self, args, vocabs, type2vec):
        hidden_size = 300
        self.extra_features = [PoincareDistance.apply, CosineSimilarity(), polarization_identity, euclidean_dot_product]
        self.input_size = args.type_dims + args.neighbors * (args.type_dims + len(self.extra_features))
        self.type_quantity = len(type2vec)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(Classifier, self).__init__()
        self.type_lut = nn.Embedding(
            vocabs[TYPE_VOCAB].size(),
            args.type_dims
        )
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = False
        self.W1 = nn.Linear(self.input_size, hidden_size, bias=args.bias == 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.W2 = nn.Linear(hidden_size, args.neighbors, bias=args.bias == 1)
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
        neighbor_embeds = embeds.view(embeds.size(0) * embeds.size(1), -1)

        extra_features = self.get_extra_features(type_embeddings, neighbor_embeds, neighbor_indexes.size(1))

        neighbor_representation = torch.cat((neighbor_embeds, extra_features), dim=1)
        neighbor_representation = neighbor_representation.view(len(type_embeddings), -1)

        input = torch.cat((type_embeddings, neighbor_representation), dim=1).to(self.device)

        layer_one = self.dropout(self.relu(self.W1(input)))
        layer_two = self.W2(layer_one)
        distribution = self.sg(layer_two)

        loss = self.loss_func(layer_two, one_hot_neighbor_types) if one_hot_neighbor_types is not None else None
        return distribution, loss

    def get_extra_features(self, predictions, true_types, amount_of_neighbors):
        expanded_predictions = expand_tensor(predictions, amount_of_neighbors)

        result = self.extra_features[0](expanded_predictions, true_types).unsqueeze(1)
        for f in self.extra_features[1:]:
            partial = f(expanded_predictions, true_types)
            result = torch.cat((result, partial.unsqueeze(1)), dim=1)

        return result


def euclidean_dot_product(u, v):
    return (u * v).sum(dim=1)
