#!/usr/bin/env python
# encoding: utf-8

import copy
import torch
from torch.nn.utils import clip_grad_norm_
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

        self.result_printer.show()

        self.validate_set(self.dev_data, "dev")

        self.validate_projection(self.test_data, "test", plot=True)
        test_results, test_eval = self.validate_set(self.test_data, "test")

        return raw_evaluate(test_results), test_eval, None

    def validate_set(self, dataset, name):
        log.info(f"\n\n\nVALIDATION ON {name.upper()}")
        _, set_results = self.validate(dataset)
        eval_result = evaluate(set_results)
        # stratified_dev_eval, _ = stratified_evaluate(dev_results, self.vocabs[TYPE_VOCAB])
        log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)\n" + eval_result)
        # log.info("Final Stratified evaluation on DEV:\n" + stratified_dev_eval)
        return set_results, eval_result

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_model_loss, total_classif_loss, total_angles, total_pos_dist, total_euclid_dist, total_norms = [], [], [], [], [], []
        self.set_learning_rate(epoch)
        self.model.train()
        self.classifier.train()
        for i in tqdm(range(niter), desc="train_epoch_{}".format(epoch)):
            batch = self.train_data[i]
            types = batch[5]

            self.model_optim.zero_grad()
            model_loss, type_embeddings, feature_repre, _, angles, dist_to_pos, euclid_dist = self.model(batch, epoch)
            model_loss.backward(retain_graph=True)
            if self.args.max_grad_norm >= 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.model_optim.step()

            neighbor_indexes, one_hot_neighbor_types = self.knn.neighbors(type_embeddings, types, self.args.neighbors)

            self.classifier_optim.zero_grad()
            _, classifier_loss = self.classifier(type_embeddings, neighbor_indexes, feature_repre, one_hot_neighbor_types)
            classifier_loss.backward()
            if self.args.max_grad_norm >= 0:
                clip_grad_norm_(self.classifier.parameters(), self.args.max_grad_norm)
            self.classifier_optim.step()

            # Stats
            total_angles.append(angles)
            total_pos_dist.append(dist_to_pos)
            total_euclid_dist.append(euclid_dist)
            total_norms.append(torch.norm(type_embeddings, p=2, dim=1))
            total_model_loss.append(model_loss.item())
            total_classif_loss.append(classifier_loss.item())

        all_pos = torch.cat(total_pos_dist)
        all_euclid = torch.cat(total_euclid_dist)
        all_angles = torch.cat(total_angles)
        all_pred_norm = torch.cat(total_norms)

        log.debug(f"Train epoch {epoch}: d to pos: {all_pos.mean():0.2f} +- {all_pos.std():0.2f}, "
                  f"Euclid dist: {all_euclid.mean():0.2f} +- {all_euclid.std():0.2f}, "
                  f"Angles: {all_angles.mean():0.2f} +- {all_angles.std():0.2f}, "
                  f"Norm:{all_pred_norm.mean():0.2f} +- {all_pred_norm.std():0.2f}\n"
                  f"cos_fact:{self.args.cosine_factor}, norm_fact:{self.args.norm_factor}, "
                  f"max norm:{all_pred_norm.max().item()}, min norm:{all_pred_norm.min().item()}")
        return np.mean(total_model_loss), np.mean(total_classif_loss)

    def validate_projection(self, data, name, epoch=None, plot=False):
        total_model_loss, total_pos_dist, total_euclid_dist, total_norms, total_angles = [], [], [], [], []
        full_type_positions, full_closest_true_neighbor = [], []

        log.info(f"Validating projection on {name.upper()} data")

        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                types = batch[5]

                model_loss, type_embeddings, feature_repre, _, angles, dist_to_pos, euclid_dist = self.model(batch, 0)

                total_pos_dist.append(dist_to_pos)
                total_euclid_dist.append(euclid_dist)
                total_norms.append(torch.norm(type_embeddings, p=2, dim=1))
                total_angles.append(angles)

                total_model_loss.append(model_loss.item())

                type_positions, closest_true_neighbor = self.knn.type_positions(type_embeddings, types)
                full_type_positions.extend(type_positions)
                full_closest_true_neighbor.extend(closest_true_neighbor)

            self.log_neighbor_positions(full_closest_true_neighbor, "CLOSEST", self.args.neighbors)
            self.log_neighbor_positions(full_type_positions, "FULL", self.args.neighbors)

            all_pos = torch.cat(total_pos_dist)
            all_euclid = torch.cat(total_euclid_dist)
            all_pred_norm = torch.cat(total_norms)
            all_angles = torch.cat(total_angles)

            all_euclid_mean = all_euclid.mean()

            if plot:
                plot_k(name, full_type_positions, full_closest_true_neighbor)

            log.debug(f"\nProj {name.upper()} epoch {epoch}: d to pos: {all_pos.mean():0.2f} +- {all_pos.std():0.2f}, "
                      f"Euclid: {all_euclid_mean:0.2f} +- {all_euclid.std():0.2f}, "
                      f"Angles: {all_angles.mean():0.2f} +- {all_angles.std():0.2f}, "
                      f"Mean norm:{all_pred_norm.mean():0.2f}+-{all_pred_norm.std():0.2f}")
            self.log_config()

            return all_euclid_mean

    def validate(self, data, epoch=None):
        total_model_loss, total_classif_loss = [], []
        results = []
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            for i in range(len(data)):
                batch = data[i]
                types = batch[5]

                model_loss, predicted_embeds, feature_repre, _, _, _, _ = self.model(batch, epoch)

                neighbor_indexes, one_hot_neighbor_types = self.knn.neighbors(predicted_embeds, types, self.args.neighbors)

                predictions, classifier_loss = self.classifier(predicted_embeds, neighbor_indexes, feature_repre, one_hot_neighbor_types)

                total_model_loss.append(model_loss.item())
                total_classif_loss.append(classifier_loss.item())

                results += assign_types(predictions, neighbor_indexes, types, self.hierarchy)

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
