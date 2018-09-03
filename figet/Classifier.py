
import torch
import torch.nn as nn
from figet.utils import expand_tensor
from figet.Constants import TYPE_VOCAB


class Classifier(nn.Module):
    def __init__(self, args, vocabs, type2vec):
        extra_features = 0
        hidden_size = 300
        self.input_size = args.type_dims + args.neighbors * (args.type_dims + extra_features)
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

        # self.softmax = nn.Softmax(dim=0)
        # self.loss_func = nn.CrossEntropyLoss()

    def forward(self, type_embeddings, neighbor_indexes, one_hot_neighbor_types=None):
        """
        :param type_embeddings: batch x type_dim
        :param neighbor_indexes: batch x k
        :param one_hot_neighbor_types: batch x k
        :return:
        """
        embeds = self.type_lut(neighbor_indexes)

        neighbor_embeds = embeds.view(-1, embeds.size(1) * embeds.size(2))

        input = torch.cat((type_embeddings, neighbor_embeds), dim=1).to(self.device)

        layer_one = self.dropout(self.relu(self.W1(input)))
        layer_two = self.W2(layer_one)
        distribution = self.sg(layer_two)

        loss = self.loss_func(layer_two, one_hot_neighbor_types) if one_hot_neighbor_types is not None else None
        return distribution, loss
