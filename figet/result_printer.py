import torch
from operator import itemgetter
from figet.utils import get_logging
from figet.Constants import TOKEN_VOCAB, TYPE_VOCAB, COARSE_FLAG, FINE_FLAG, UF_FLAG
from figet.Predictor import assign_types, assign_total_types
from figet.evaluate import COARSE, FINE

log = get_logging()

ASSIGN = 0
TRUE = 1
CORRECT = 2


def stratify(types, co_fi_ids):
    co_fi, uf = set(), set()
    for t in types.tolist():
        if t in co_fi_ids:
            co_fi.add(t)
        else:
            uf.add(t)
    return co_fi, uf


class ResultPrinter(object):

    def __init__(self, test_data, vocabs, model, classifier, knn, hierarchy, args):
        self.data = test_data
        self.token_vocab = vocabs[TOKEN_VOCAB]
        self.type_vocab = vocabs[TYPE_VOCAB]
        self.model = model
        self.knn = knn
        self.hierarchy = hierarchy
        self.args = args
        self.grans = [COARSE_FLAG, FINE_FLAG, UF_FLAG]
        self.coarse_matrixes = [{self.type_vocab.label2idx[label]: [0, 0, 0] for label in COARSE
                                 if label in self.type_vocab.label2idx} for _ in self.grans]
        self.coarse_ids = [self.type_vocab.label2idx[label] for label in COARSE if label in self.type_vocab.label2idx]
        self.fine_ids = [self.type_vocab.label2idx[label] for label in FINE if label in self.type_vocab.label2idx]
        self.co_fi_ids = set(self.coarse_ids + self.fine_ids)

    def show(self, n=2):
        to_show = []
        with torch.no_grad():
            for batch_index in range(len(self.data)):
                batch = self.data[batch_index]
                types = batch[5]

                model_loss, predicted_embeds, feature_repre, attn, _, _, _ = self.model(batch, self.args.epochs)
                # neighbor_indexes = [self.knn.neighbors(pred, -1, gran_id) for gran_id, pred in enumerate(type_embeddings)]

                partial_result = assign_total_types(predicted_embeds, types, self.knn)

                for i in range(len(partial_result)):
                    true, predicted = partial_result[i]
                    corrects = len(set(predicted.tolist()).intersection(set(true.tolist())))
                    accuracy = corrects / len(true)
                    # accuracy, mention_idx, ctx, attn, true, predicted, neighbors
                    to_show.append([accuracy, batch[3][i], batch[0][i], attn[i].tolist(), true, predicted])

                # self.update_coarse_matrixes(partial_result)

        self.print_results(to_show)

        # self.print_coarse_matrix()

    def print_results(self, to_show):
        unk = "@"
        to_show = sorted(to_show, key=itemgetter(0), reverse=True)
        for accuracy, mention, ctx, attn, true, predicted in to_show:
            mention_words = " ".join([self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in mention if i != 0])

            ctx_words = [self.token_vocab.get_label_from_word2vec_id(i.item(), unk) for i in ctx]
            ctx_and_attn = map(lambda t: t[0] + f"({t[1][0]:0.2f})", zip(ctx_words, attn))
            ctx_words = " ".join(ctx_and_attn)

            true_co_fi, true_uf = stratify(true, self.co_fi_ids)
            pred_co_fi, pred_uf = stratify(predicted, self.co_fi_ids)

            true_co_fi_types = " ".join([self.type_vocab.get_label(i) for i in true_co_fi])
            true_uf_types = " ".join([self.type_vocab.get_label(i) for i in true_uf])
            pred_co_fi_types = " ".join([self.type_vocab.get_label(i) for i in pred_co_fi])
            pred_uf_types = " ".join([self.type_vocab.get_label(i) for i in pred_uf])
            # neighbor_types = " ".join([self.type_vocab.get_label(i.item()) for i in neighbors])

            log.debug(f"Mention: '{mention_words}'\nCtx:'{ctx_words}'\n"
                      f"Acc: {accuracy * 100:0.2f}: True: co: '{true_co_fi_types}', uf: '{true_uf_types}' - "
                      f"Pred: co:'{pred_co_fi_types}', uf: '{pred_uf_types}'\n\n")

    def update_coarse_matrixes(self, results):
        for idx in range(len(results)):
            result = results[idx]
            matrix = self.coarse_matrixes[idx]

            for true_types, predictions in result:
                true_set = set([x.item() for x in true_types])
                for true_type in true_set:
                    if true_type in matrix:
                        matrix[true_type][TRUE] += 1

                for predicted in [y.item() for y in predictions]:
                    if predicted in matrix:
                        matrix[predicted][ASSIGN] += 1

                        if predicted in true_set:
                            matrix[predicted][CORRECT] += 1

    def print_coarse_matrix(self):
        grans = ["COARSE", "FINE", "ULTRAFINE"]
        for i in range(len(self.coarse_matrixes)):
            matrix = self.coarse_matrixes[i]
            results = []
            for coarse, values in matrix.items():
                label = self.type_vocab.get_label(coarse)
                assign, true, correct = values[ASSIGN], values[TRUE], values[CORRECT]
                p = correct / assign * 100 if assign != 0 else 0
                r = correct / true * 100 if true != 0 else 0
                f1 = 2 * p * r / (p + r) if p + r != 0 else 0
                extra_tab = '    ' if label != 'organization' and label != 'location' else ''
                results.append(f"{label}\t{extra_tab}{assign}/{correct}/{true}\t"
                               f"{p:0.2f}\t{r:0.2f}\t{f1:0.2f}")

            log.info(f"{grans[i]} labels matrix results (assign/correct/true) (P,R,F1):\n" + "\n".join(results))


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
