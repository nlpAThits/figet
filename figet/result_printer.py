import torch
from figet.utils import get_logging
from figet.Constants import TOKEN_VOCAB, TYPE_VOCAB
from figet.Predictor import assign_types
from figet.evaluate import COARSE

log = get_logging()

ASSIGN = 0
TRUE = 1
CORRECT = 2


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
        self.coarse_matrix = {self.type_vocab.label2idx[label]: [0, 0, 0] for label in COARSE
                              if label in self.type_vocab.label2idx}

    def show(self, n=2):
        filters = [is_partially_right, is_totally_wrong]
        collected = [[[], [], []],
                     [[], [], []]]
        with torch.no_grad():
            for batch_index in range(len(self.data)):
                batch = self.data[batch_index]
                types = batch[5]

                model_loss, type_embeddings, feature_repre, attn, _, _, _ = self.model(batch, self.args.epochs)
                neighbor_indexes = [self.knn.neighbors(pred, -1, gran_id)
                                    for gran_id, pred in enumerate(type_embeddings)]

                results = [assign_types(None, neighs, types, self.hierarchy) for neighs in neighbor_indexes]

                for i in range(len(filters)):
                    criteria = filters[i]
                    for gran_id, gran_result in enumerate(results):
                        to_show = []
                        for j in range(len(gran_result)):
                            true, predicted = gran_result[j]
                            if criteria(true, predicted):
                                # mention_idx, ctx, attn, true, predicted, neighbors
                                to_show.append([batch[3][j], batch[0][j], attn[j].tolist(), true, predicted, neighbor_indexes[gran_id][j][:5]])
                            if len(to_show) == n: break

                        collected[i][gran_id] += to_show

                self.update_coarse_matrix(results[0])

        filter_titles = ["+ Partially right:", "-- Totally wrong:"]
        gran_titles = ["COARSE", "FINE", "ULTRAFINE"]
        for j in range(len(gran_titles)):
            for i in range(len(filters)):
                log.debug(f"{gran_titles[j]} - {filter_titles[i]}")
                self.print_results(collected[i][j])

        self.print_coarse_matrix()

    def print_results(self, to_show):
        unk = "@"
        for mention, ctx, attn, true, predicted, neighbors in to_show:
            mention_words = " ".join([self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in mention if i != 0])

            ctx_words = [self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in ctx]
            ctx_and_attn = map(lambda t: t[0] + f"({t[1][0]:0.2f})", zip(ctx_words, attn))
            ctx_words = " ".join(ctx_and_attn)

            true_types = " ".join([self.type_vocab.get_label(i.item()) for i in true])
            predicted_types = " ".join([self.type_vocab.get_label(i.item()) for i in predicted])
            neighbor_types = " ".join([self.type_vocab.get_label(i.item()) for i in neighbors])

            log.debug(f"Mention: '{mention_words}'\nCtx:'{ctx_words}'\n"
                      f"True: '{true_types}' - Predicted: {predicted_types}\n"
                      f"Closest neighbors: {neighbor_types}\n*****")

    def update_coarse_matrix(self, results):
        for true_types, predictions in results:
            true_set = set([x.item() for x in true_types])
            for true_type in true_set:
                if true_type in self.coarse_matrix:
                    self.coarse_matrix[true_type][TRUE] += 1

            for predicted in [y.item() for y in predictions]:
                if predicted in self.coarse_matrix:
                    self.coarse_matrix[predicted][ASSIGN] += 1

                    if predicted in true_set:
                        self.coarse_matrix[predicted][CORRECT] += 1

    def print_coarse_matrix(self):
        results = []
        for coarse, values in self.coarse_matrix.items():
            label = self.type_vocab.get_label(coarse)
            assign, true, correct = values[ASSIGN], values[TRUE], values[CORRECT]
            p = correct / assign * 100 if assign != 0 else 0
            r = correct / true * 100 if true != 0 else 0
            f1 = 2 * p * r / (p + r) if p + r != 0 else 0
            results.append(f"{label}\t\t\t{assign}/{correct}/{true}\t"
                           f"{p:0.2f}\t{r:0.2f}\t{f1:0.2f}")

        log.info("COARSE labels matrix results (assign/correct/true) (P,R,F1):\n" + "\n".join(results))


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
