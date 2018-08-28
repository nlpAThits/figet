
import torch
import torch.nn as nn
from figet.utils import expand_tensor


class Classifier(nn.Module):
    def __init__(self, args, type2vec):
        self.input_size = args.classifier_input_size
        self.type2vec = type2vec.cuda() if torch.cuda.is_available() else type2vec
        super(Classifier, self).__init__()
        self.W = nn.Linear(self.input_size, 1, bias=args.bias == 1)
        self.sg = nn.Sigmoid()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, type_embeddings, truth=None):
        """
        :param type_embeddings: batch x type_dim
        :param truth: batch x true_type_len
        :return:
        """
        expanded_preds = expand_tensor(type_embeddings, len(self.type2vec))
        true_embeddings = self.type2vec.repeat(len(type_embeddings), 1)
        input = torch.cat((expanded_preds, true_embeddings), dim=1)

        output = self.W(input)
        output = self.sg(output)
        loss = self.loss_func(output, truth.view(truth.size(0) * truth.size(1), 1)) if truth is not None else None
        return output, loss
