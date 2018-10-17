#!/usr/bin/env python
# encoding: utf-8

import time
import copy
import torch
import numpy as np
from tqdm import tqdm

from figet.utils import get_logging, plot_k
from figet.Predictor import kNN, assign_types
from figet.evaluate import evaluate, raw_evaluate, stratified_evaluate
from figet.Constants import TYPE_VOCAB
from figet.result_printer import ResultPrinter

log = get_logging()


class Coach(object):

    def __init__(self, model, optim, classifier, classifier_optim, vocabs, train_data, dev_data, test_data, hard_test_data, type2vec, word2vec, hierarchy, args, extra_args, config):
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
        self.config = config

    def train(self):
        log.debug(self.model)
        log.debug(self.classifier)

        self.start_time = time.time()
        train_subsample = self.train_data.subsample(2000)

        for epoch in range(1, self.args.epochs + 1):
            train_model_loss, train_classif_loss = self.train_epoch(epoch)

            if epoch == self.args.epochs:
                log.info("\n\n------FINAL RESULTS----------")

            self.validate_projection(train_subsample, "train", epoch)
            self.validate_projection(self.dev_data, "dev", epoch)

            log.info(f"Results epoch {epoch}: "
                     f"TRAIN loss: model: {train_model_loss:.2f}, classif:{train_classif_loss:.2f}")

        # self.result_printer.show()
        self.validate_projection(self.test_data, "test")
        test_loss, test_results = self.validate(self.test_data, show_positions=True)
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
        total_model_loss, total_classif_loss, total_avg_target_norm, total_pos_dist, total_euclid_dist, total_norms = [], [], [], [], [], []
        self.model.train()
        self.classifier.train()
        for i in tqdm(range(niter), desc="train_epoch_{}".format(epoch)):
            batch = self.train_data[i]

            self.model_optim.zero_grad()
            model_loss, type_embeddings, _, avg_target_norm, dist_to_pos, euclid_dist = self.model(batch, epoch)
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
            total_euclid_dist.append(euclid_dist)
            total_norms.append(torch.norm(type_embeddings, p=2, dim=1))
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
                log.debug(f"Mean batch norm: {mean_norm:0.2f}, max norm: {max_norm}, min norm: {min_norm}")

        all_pos = torch.cat(total_pos_dist)
        all_euclid = torch.cat(total_euclid_dist)
        all_avg_target_norm = torch.cat(total_avg_target_norm)
        all_pred_norm = torch.cat(total_norms)

        log.debug(f"AVGS:\nd to pos: {all_pos.mean():0.2f} +- {all_pos.std():0.2f}, Euclid distance: {all_euclid.mean():0.2f} +- {all_euclid.std():0.2f} "
                  f"cos_fact:{self.args.cosine_factor}, norm_fact:{self.args.norm_factor}\n"
                  f"Mean norm:{all_pred_norm.mean():0.2f}, max norm:{all_pred_norm.max().item()}, min norm:{all_pred_norm.min().item()}")
        return np.mean(total_model_loss), np.mean(total_classif_loss)

    def validate_projection(self, data, name, epoch=None):
        total_model_loss, total_pos_dist, total_euclid_dist, total_norms = [], [], [], []
        among_top_k, total = 0, 0
        full_type_positions, full_closest_true_neighbor = [], []

        log.info(f"Validating projection on {name.upper()} data")

        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                types = batch[5]

                model_loss, type_embeddings, _, avg_target_norm, dist_to_pos, euclid_dist = self.model(batch, 0)

                total_pos_dist.append(dist_to_pos)
                total_euclid_dist.append(euclid_dist)
                total_norms.append(torch.norm(type_embeddings, p=2, dim=1))

                total_model_loss.append(model_loss.item())
                total += len(types)

                type_positions, closest_true_neighbor = self.knn.type_positions(type_embeddings, types)
                full_type_positions.extend(type_positions)
                full_closest_true_neighbor.extend(closest_true_neighbor)

            self.log_neighbor_positions(full_closest_true_neighbor, "CLOSEST", self.args.neighbors)
            self.log_neighbor_positions(full_type_positions, "FULL", self.args.neighbors)
            log.info("Precision@{}: {:.2f}".format(self.args.neighbors, float(among_top_k) * 100 / total))

            all_pos = torch.cat(total_pos_dist)
            all_euclid = torch.cat(total_euclid_dist)
            all_pred_norm = torch.cat(total_norms)

            log.debug(f"\nProj {name.upper()} epoch {epoch}: d to pos: {all_pos.mean():0.2f}+-{all_pos.std():0.2f}, "
                      f"Euclid: {all_euclid.mean():0.2f}+-{all_euclid.std():0.2f}, "
                      f"Mean norm:{all_pred_norm.mean():0.2f}+-{all_pred_norm.std():0.2f}")
            self.log_config()

            return np.mean(total_model_loss)

    def validate(self, data, show_positions=False, epoch=None):
        total_model_loss, total_classif_loss = [], []
        results = []
        full_type_positions, full_closest_true_neighbor = [], []
        among_top_k, total = 0, 0
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                types = batch[5]

                model_loss, type_embeddings, _, _, _, _ = self.model(batch, epoch)

                neighbor_indexes, one_hot_neighbor_types = self.knn.neighbors(type_embeddings, types, self.args.neighbors)

                predictions, classifier_loss = self.classifier(type_embeddings, neighbor_indexes, one_hot_neighbor_types)

                total_model_loss.append(model_loss.item())
                total_classif_loss.append(classifier_loss.item())

                results += assign_types(predictions, neighbor_indexes, types, self.hierarchy)

                among_top_k += self.knn.precision_at(type_embeddings, types, k=self.args.neighbors)
                total += len(types)

                if show_positions:
                    type_positions, closest_true_neighbor = self.knn.type_positions(type_embeddings, types)
                    full_type_positions.extend(type_positions)
                    full_closest_true_neighbor.extend(closest_true_neighbor)

            if show_positions:
                self.log_neighbor_positions(full_closest_true_neighbor, "CLOSEST", self.args.neighbors)
                self.log_neighbor_positions(full_type_positions, "FULL", self.args.neighbors)

                plot_k(full_type_positions, full_closest_true_neighbor)

            log.info("Precision@{}: {:.2f}".format(self.args.neighbors, float(among_top_k) * 100 / total))

            return np.mean(total_model_loss) + np.mean(total_classif_loss), results

    def log_neighbor_positions(self, positions, name, k):
        log.info("{} neighbor positions: Mean:{:.2f} Std: {:.2f}".format(name, np.mean(positions), np.std(positions)))
        self.log_proportion(k // 2, positions)
        self.log_proportion(k, positions)
        self.log_proportion(3 * k // 2, positions)

    def log_proportion(self, k, positions):
        proportion = sum(val < k for val in positions) / float(len(positions)) * 100
        log.info("Proportion of neighbors in first {}: {:.2f}%".format(k, proportion))

    def log_config(self):
        config = self.config
        log.info(f"cosine_factor:{config[11]}, norm_factor:{config[12]}, hyperdist_factor:{config[13]}")
