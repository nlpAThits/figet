from figet.utils import get_logging
from figet.Constants import TOKEN_VOCAB, TYPE_VOCAB
from figet.Predictor import assign_types
import torch
import numpy as np

log = get_logging()


class ResultPrinter(object):

    def __init__(self, test_data, vocabs, model, classifier, knn, hierarchy, args):
        self.data = test_data
        self.token_vocab = vocabs[TOKEN_VOCAB]
        self.type_vocab = vocabs[TYPE_VOCAB]
        self.model = model
        self.classifier = classifier
        self.knn = knn
        self.hierarchy = hierarchy
        self.args = args

    def show(self, n=2):
        filters = [is_strictly_right, is_partially_right, is_totally_wrong]
        collected = [[], [], []]

        for batch_index in range(len(self.data)):
            batch = self.data[batch_index]
            types = batch[3]

            model_loss, type_embeddings, _, _, _, _ = self.model(batch, self.args.epochs)
            neighbor_indexes, one_hot_neighbor_types = self.knn.neighbors(type_embeddings, types, self.args.neighbors)
            predictions, _ = self.classifier(type_embeddings, neighbor_indexes, one_hot_neighbor_types)

            results = assign_types(predictions, neighbor_indexes, types, self.hierarchy)

            for i in range(len(filters)):
                criteria = filters[i]
                to_show = []
                for j in range(len(results)):
                    true, predicted = results[j]
                    if criteria(true, predicted):
                        # mention_idx, prev_ctx, next_ctx, true, predicted, neighbors
                        to_show.append([batch[4][j], batch[1][j], batch[2][j], true, predicted, neighbor_indexes[j][:3]])
                    if len(to_show) == n: break

                collected[i] += to_show

        log_titles = ["\n\n++ Strictly right:", "\n\n+ Partially right:", "\n\n-- Totally wrong:"]
        for i in range(len(filters)):
            log.debug(log_titles[i])
            self.print_results(collected[i])

    def print_results(self, to_show):
        unk = "@"
        for mention, prev_ctx, next_ctx, true, predicted, neighbors in to_show:
            mention_words = " ".join([self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in mention if i != 0])
            prev_words = " ".join([self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in prev_ctx])
            next_words = " ".join([self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in np.flip(next_ctx.cpu().numpy(), 0)])

            true_types = " ".join([self.type_vocab.get_label(i.item()) for i in true])
            predicted_types = " ".join([self.type_vocab.get_label(i.item()) for i in predicted])
            neighbor_types = " ".join([self.type_vocab.get_label(i.item()) for i in neighbors])

            log.debug(f"Left: '{prev_words}'\nMention: '{mention_words}'\nRight:'{next_words}'\n"
                      f"True: '{true_types}' - Predicted: {predicted_types}\n"
                      f"Closest neighbors: {neighbor_types}\n*****")


def is_strictly_right(true, predicted):
    if true.size() != predicted.size():
        return False
    return torch.all(true == predicted).item() == 1


def is_partially_right(true, predicted):
    rights = len(set([i.item() for i in predicted]).intersection(set([j.item() for j in true])))
    return 0 < rights < len(true)


def is_totally_wrong(true, predicted):
    rights = len(set([i.item() for i in predicted]).intersection(set([j.item() for j in true])))
    return rights == 0
