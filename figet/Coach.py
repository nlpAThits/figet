#!/usr/bin/env python
# encoding: utf-8

import time
import copy
import torch
import numpy as np
from tqdm import tqdm

from figet.utils import get_logging
from figet.Predictor import kNN, assign_types
from figet.evaluate import evaluate, raw_evaluate, stratified_evaluate
from figet.Constants import TYPE_VOCAB
from figet.result_printer import ResultPrinter

log = get_logging()


class Coach(object):

    def __init__(self, model, optim, classifier, classifier_optim, vocabs, train_data, dev_data, test_data, hard_test_data, type2vec, word2vec, hierarchy, args, extra_args):
        self.model = model
        self.model_optim = optim
        self.classifier = classifier
        self.classifier_optim = classifier_optim
        self.vocabs = vocabs
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.hard_test_data = hard_test_data
        self.hierarchy = hierarchy
        self.args = args
        self.word2vec = word2vec
        self.type2vec = type2vec
        self.knn = kNN(vocabs[TYPE_VOCAB], type2vec, extra_args["knn_metric"] if extra_args["knn_metric"] else None)
        self.result_printer = ResultPrinter(test_data, vocabs, model, classifier, self.knn, hierarchy, args)

    def train(self):
        log.debug(self.model)
        log.debug(self.classifier)

        # Early stopping
        best_dev_macro_f1, best_epoch, best_model_state, best_classif_state = -1, 0, None, None

        self.start_time = time.time()
        train_subsample = self.train_data.subsample(2000)

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)

            if epoch == self.args.epochs:
                log.info("\n\n------FINAL RESULTS----------")

            log.info("Validating on TRAIN data")
            _, train_results = self.validate(train_subsample, epoch == self.args.epochs, epoch)
            train_eval = evaluate(train_results)
            log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)\n" + train_eval)

            log.info("Validating on DEV data")
            dev_loss, dev_results = self.validate(self.dev_data, epoch == self.args.epochs, epoch)
            dev_eval = evaluate(dev_results)
            stratified_dev_eval = stratified_evaluate(dev_results, self.vocabs[TYPE_VOCAB])

            log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)\n" + dev_eval)
            log.info("Stratified evaluation:\n" + stratified_dev_eval)

            log.info("Results epoch {}: Model TRAIN loss: {:.2f}. DEV loss: {:.2f}".format(epoch, train_loss, dev_loss))

            dev_macro_f1 = float(dev_eval.split("\t")[5])
            if dev_macro_f1 > best_dev_macro_f1:
                best_dev_macro_f1 = dev_macro_f1
                best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_classif_state = copy.deepcopy(self.classifier.state_dict())
                log.info(f"Best DEV F1 found at epoch{epoch}")
                log.info(dev_eval)

        # log.info(f"\nFINAL: Evaluating on TEST data with best state from epoch: {best_epoch}")
        # self.model.load_state_dict(best_model_state)
        # self.classifier.load_state_dict(best_classif_state)

        self.result_printer.show()

        test_loss, test_results = self.validate(self.test_data, True, None)
        test_eval = evaluate(test_results)
        stratified_test_eval = stratified_evaluate(test_results, self.vocabs[TYPE_VOCAB])
        log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)\n" + test_eval)
        log.info("Final Stratified evaluation on test:\n" + stratified_test_eval)

        return raw_evaluate(test_results), test_eval, stratified_test_eval

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_model_loss, total_classif_loss, total_avg_target_norm, total_pos_dist, total_neg_dist = [], [], [], [], []
        self.model.train()
        self.classifier.train()
        for i in tqdm(range(niter), desc="train_epoch_{}".format(epoch)):
            batch = self.train_data[i]

            self.model_optim.zero_grad()
            model_loss, type_embeddings, _, avg_target_norm, dist_to_pos, dist_to_neg = self.model(batch, epoch)
            model_loss.backward(retain_graph=True)
            self.model_optim.step()

            neighbor_indexes, one_hot_neighbor_types = self.knn.neighbors(type_embeddings, batch[3], self.args.neighbors)

            self.classifier_optim.zero_grad()
            _, classifier_loss = self.classifier(type_embeddings, neighbor_indexes, one_hot_neighbor_types)
            classifier_loss.backward()
            self.classifier_optim.step()

            # Stats.
            total_avg_target_norm.append(avg_target_norm)
            total_pos_dist.append(dist_to_pos)
            total_neg_dist.append(dist_to_neg)

            total_model_loss.append(model_loss.item())
            total_classif_loss.append(classifier_loss.item())

            if (i + 1) % self.args.log_interval == 0:
                norms = torch.norm(type_embeddings, p=2, dim=1)
                mean_norm = norms.mean().item()
                max_norm = norms.max().item()
                min_norm = norms.min().item()
                avg_model_loss, avg_classif_loss = np.mean(total_model_loss), np.mean(total_classif_loss)

                log.debug("Epoch %2d | %5d/%5d | loss %6.4f | %6.0f s elapsed"
                    % (epoch, i+1, len(self.train_data), avg_model_loss + avg_classif_loss, time.time()-self.start_time))
                log.debug(f"Model loss: {avg_model_loss}, Classif loss: {avg_classif_loss}")
                log.debug(f"Mean norm: {mean_norm:0.2f}, max norm: {max_norm}, min norm: {min_norm}")

        all_pos = torch.cat(total_pos_dist)
        all_neg = torch.cat(total_neg_dist)
        all_avg_target_norm = torch.cat(total_avg_target_norm)

        log.debug(f"AVGS: \nTotal avg target norm: {all_avg_target_norm.mean():0.3f} +- {all_avg_target_norm.std():0.3f}, "
                  f"d to pos: {all_pos.mean():0.2f} +- {all_pos.std():0.2f}, d to neg: {all_neg.mean():0.2f} +- {all_neg.std():0.2f}")
        return np.mean(total_model_loss) # + np.mean(total_classif_loss)

    def validate(self, data, show_positions=False, epoch=None):
        total_model_loss, total_classif_loss = [], []
        results = []
        true_positions = []
        k = int(self.args.neighbors / 2)
        among_top_k, total = 0, 0
        self.model.eval()
        self.classifier.eval()
        for i in range(len(data)):
            batch = data[i]
            types = batch[3]

            model_loss, type_embeddings, _, _, _, _ = self.model(batch, epoch)

            neighbor_indexes, one_hot_neighbor_types = self.knn.neighbors(type_embeddings, types, self.args.neighbors)

            predictions, classifier_loss = self.classifier(type_embeddings, neighbor_indexes, one_hot_neighbor_types)

            total_model_loss.append(model_loss.item())
            total_classif_loss.append(classifier_loss.item())

            results += assign_types(predictions, neighbor_indexes, types, self.hierarchy)

            among_top_k += self.knn.precision_at(type_embeddings, types, k=self.args.neighbors)
            total += len(types)

            if show_positions:
                true_positions.extend(self.knn.true_types_position(type_embeddings, types))

        if show_positions:
            log.info("Positions: Mean:{:.2f} Std: {:.2f}".format(np.mean(true_positions), np.std(true_positions)))
            proportion = sum(val < k for val in true_positions) / float(len(true_positions)) * 100
            log.info("Proportion of neighbors in first {}: {}".format(k, proportion))
            proportion = sum(val < self.args.neighbors for val in true_positions) / float(len(true_positions)) * 100
            log.info("Proportion of neighbors in first {}: {}".format(self.args.neighbors, proportion))
            proportion = sum(val < 3 * k for val in true_positions) / float(len(true_positions)) * 100
            log.info("Proportion of neighbors in first {}: {}".format(3 * k, proportion))

        log.info("Precision@{}: {:.2f}".format(self.args.neighbors, float(among_top_k) * 100 / total))

        return np.mean(total_model_loss) + np.mean(total_classif_loss), results
