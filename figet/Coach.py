#!/usr/bin/env python
# encoding: utf-8

import copy
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from statistics import mean, stdev, median, mode, StatisticsError

from figet.utils import get_logging, plot_k
from figet.Predictor import kNN, assign_types
from figet.evaluate import evaluate, raw_evaluate, stratified_evaluate
from figet.Constants import TYPE_VOCAB, COARSE_FLAG, FINE_FLAG, UF_FLAG
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
        self.knn = kNN(type2vec, vocabs[TYPE_VOCAB], args.knn_hyper)
        self.result_printer = ResultPrinter(dev_data, vocabs, model, classifier, self.knn, hierarchy, args)
        self.config = config
        self.granularities = [COARSE_FLAG, FINE_FLAG, UF_FLAG]

    def train(self):
        log.debug(self.model)
        log.debug(self.classifier)

        min_euclid_dist, best_model_state, best_epoch = 100, None, 0

        for epoch in range(1, self.args.epochs + 1):
            train_model_loss, train_classif_loss = self.train_epoch(epoch)

            euclid_dist = self.validate_projection(self.dev_data, "dev", epoch, plot=epoch == self.args.epochs)

            log.info(f"Results epoch {epoch}: "
                     f"TRAIN loss: model: {train_model_loss:.2f}, classif:{train_classif_loss:.5f}")

            if euclid_dist < min_euclid_dist:
                min_euclid_dist = euclid_dist
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                log.info(f"* Best euclid dist {min_euclid_dist:0.2f} at epoch {epoch} *")

            if epoch % 10 == 0:
                self.validate_set(self.dev_data, "dev")

        log.info(f"Final evaluation on best distance ({min_euclid_dist}) from epoch {best_epoch}")
        self.model.load_state_dict(best_model_state)

        # self.result_printer.show()

        self.validate_set(self.dev_data, "dev")

        self.validate_projection(self.test_data, "test", plot=True)
        test_results, test_eval = self.validate_set(self.test_data, "test")

        return raw_evaluate(test_results), test_eval, None

    def validate_set(self, dataset, name):
        log.info(f"\n\n\nVALIDATION ON {name.upper()}")
        _, all_results = self.validate(dataset)
        for set_results in all_results:
            eval_result = evaluate(set_results)
            stratified_dev_eval, _ = stratified_evaluate(set_results, self.vocabs[TYPE_VOCAB])
            log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)\n" + eval_result)
            log.info(f"Final Stratified evaluation on {name.upper()}:\n" + stratified_dev_eval)
        return all_results[0], eval_result

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_model_loss = []
        coarse_angles, coarse_pos_dist, coarse_euclid_dist, coarse_norms = [], [], [], []
        fine_angles, fine_pos_dist, fine_euclid_dist, fine_norms = [], [], [], []
        uf_angles, uf_pos_dist, uf_euclid_dist, uf_norms = [], [], [], []
        stats = [[coarse_angles, coarse_pos_dist, coarse_euclid_dist, coarse_norms],
                 [fine_angles, fine_pos_dist, fine_euclid_dist, fine_norms],
                 [uf_angles, uf_pos_dist, uf_euclid_dist, uf_norms]]


        self.set_learning_rate(epoch)
        self.model.train()
        self.classifier.train()
        for i in tqdm(range(niter), desc="train_epoch_{}".format(epoch)):
            batch = self.train_data[i]
            types = batch[5]

            self.model_optim.zero_grad()
            model_loss, type_embeddings, feature_repre, _, angles, dist_to_pos, euclid_dist = self.model(batch, epoch)
            # model_loss.backward(retain_graph=True)
            model_loss.backward()
            if self.args.max_grad_norm >= 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.model_optim.step()

            # neighbor_indexes, one_hot_neighbor_types = self.knn.neighbors(type_embeddings, types, self.args.neighbors)
            #
            # self.classifier_optim.zero_grad()
            # _, classifier_loss = self.classifier(type_embeddings, neighbor_indexes, feature_repre, one_hot_neighbor_types)
            # classifier_loss.backward()
            # if self.args.max_grad_norm >= 0:
            #     clip_grad_norm_(self.classifier.parameters(), self.args.max_grad_norm)
            # self.classifier_optim.step()

            # Stats.
            for idx, item in enumerate(stats):
                item[0].append(angles[idx].mean().item())
                item[1].append(dist_to_pos[idx].mean().item())
                item[2].append(euclid_dist[idx].mean().item())
                item[3].append(torch.norm(type_embeddings[idx].detach(), p=2, dim=1).mean().item())

            total_model_loss.append(model_loss.item())
            # total_classif_loss.append(classifier_loss.item())

        labels = ["COARSE", "FINE", "ULTRAFINE"]
        for idx, item in enumerate(stats):
            log.debug(f"Train epoch {epoch} {labels[idx]}: d to pos: {mean(item[1]):0.2f} +- {stdev(item[1]):0.2f}, "
                      f"Euclid dist: {mean(item[2]):0.2f} +- {stdev(item[2]):0.2f}, "
                      f"Angles: {mean(item[0]):0.2f} +- {stdev(item[0]):0.2f}, "
                      f"Norm:{mean(item[3]):0.2f} +- {stdev(item[3]):0.2f}\n"
                      f"Avg max norm:{max(item[3]):0.3f}, avg min norm:{min(item[3]):0.3f}")
        return np.mean(total_model_loss), 0

    def validate_projection(self, data, name, epoch=None, plot=False):
        total_model_loss = []
        coarse_angles, coarse_pos_dist, coarse_euclid_dist, coarse_norms = [], [], [], []
        fine_angles, fine_pos_dist, fine_euclid_dist, fine_norms = [], [], [], []
        uf_angles, uf_pos_dist, uf_euclid_dist, uf_norms = [], [], [], []
        stats = [[coarse_angles, coarse_pos_dist, coarse_euclid_dist, coarse_norms],
                 [fine_angles, fine_pos_dist, fine_euclid_dist, fine_norms],
                 [uf_angles, uf_pos_dist, uf_euclid_dist, uf_norms]]

        coarse_full_type_positions, coarse_full_closest_true_neighbor = [], []
        fine_full_type_positions, fine_full_closest_true_neighbor = [], []
        uf_full_type_positions, uf_full_closest_true_neighbor = [], []
        positions = [[coarse_full_type_positions, coarse_full_closest_true_neighbor],
                     [fine_full_type_positions, fine_full_closest_true_neighbor],
                     [uf_full_type_positions, uf_full_closest_true_neighbor]]

        log.info(f"Validating projection on {name.upper()} data")

        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                types = batch[5]

                model_loss, predicted_embeds, feature_repre, _, angles, dist_to_pos, euclid_dist = self.model(batch, 0)

                # Stats.
                for idx, item in enumerate(stats):
                    item[0].append(angles[idx].mean().item())
                    item[1].append(dist_to_pos[idx].mean().item())
                    item[2].append(euclid_dist[idx].mean().item())
                    item[3].append(torch.norm(predicted_embeds[idx].detach(), p=2, dim=1).mean().item())

                total_model_loss.append(model_loss.item())

                for gran_flag, (idx, item) in zip(self.granularities, enumerate(positions)):
                    type_positions, closest_true_neighbor = self.knn.type_positions(predicted_embeds[gran_flag], types, gran_flag)
                    item[0].extend(type_positions)
                    item[1].extend(closest_true_neighbor)

            labels = ["COARSE", "FINE", "ULTRAFINE"]
            for idx, item in enumerate(positions):
                self.log_neighbor_positions(item[1], f"{labels[idx]} CLOSEST", self.args.neighbors)
                self.log_neighbor_positions(item[0], f"{labels[idx]} FULL", self.args.neighbors)

            # if plot:
            #     plot_k(name, full_type_positions, full_closest_true_neighbor)

            for idx, item in enumerate(stats):
                log.debug(f"\nProj {name.upper()} epoch {epoch} {labels[idx]}: d to pos: {mean(item[1]):0.2f} +- {stdev(item[1]):0.2f}, "
                          f"Euclid: {mean(item[2]):0.2f} +- {stdev(item[2]):0.2f}, "
                          f"Angles: {mean(item[0]):0.2f} +- {stdev(item[0]):0.2f}, "
                          f"Mean norm:{mean(item[3]):0.2f}+-{stdev(item[3]):0.2f}")
            self.log_config()

            return mean(stats[0][2])

    def validate(self, data, epoch=None):
        total_model_loss = []
        results = [[], [], []]
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                types = batch[5]

                model_loss, predicted_embeds, feature_repre, _, _, _, _ = self.model(batch, epoch)

                neighbor_indexes = []
                for gran_flag, pred in zip(self.granularities, predicted_embeds):
                    neighbor_indexes.append(self.knn.neighbors(pred, types, self.args.neighbors, gran_flag))

                # predictions, classifier_loss = self.classifier(predicted_embeds, neighbor_indexes, feature_repre, one_hot_neighbor_types)

                total_model_loss.append(model_loss.item())
                # total_classif_loss.append(classifier_loss.item())

                for idx, neighs in enumerate(neighbor_indexes):
                    results[idx] += assign_types(None, neighs, types, self.hierarchy)

            return np.mean(total_model_loss), results

    def log_neighbor_positions(self, positions, name, k):
        try:
            mode_result = mode(positions)
        except StatisticsError:
            mode_result = "2 values"

        log.info(f"{name} neighbor positions: \nMean:{mean(positions):.2f} Std: {stdev(positions):.2f}\n"
                 f"Median: {median(positions)} (middle value to have 50% on each side)\n"
                 f"Mode: {mode_result} (value that occurs more often)")
        self.log_proportion(k // 2, positions)
        self.log_proportion(k, positions)
        self.log_proportion(3 * k // 2, positions)

    def log_proportion(self, k, positions):
        proportion = sum(val < k for val in positions) / float(len(positions)) * 100
        log.info("Proportion of neighbors in first {}: {:.2f}%".format(k, proportion))

    def log_config(self):
        config = self.config
        log.info(f"cosine_factor:{config[10]}, hyperdist_factor:{config[11]}")

    def set_learning_rate(self, epoch):
        """
        :param epoch: 1-numerated
        """
        if epoch <= 2 or epoch > int(self.args.epochs * 0.9): # first two and last few epochs
            learning_rate = self.args.proj_learning_rate / 10
        else:
            learning_rate = self.args.proj_learning_rate
        for g in self.model_optim.param_groups:
            g['lr'] = learning_rate
