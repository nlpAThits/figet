#!/usr/bin/env python
# encoding: utf-8

import time
import torch
import numpy as np
from tqdm import tqdm

from figet.utils import get_logging
from figet.Predictor import kNN, assign_types
from figet.evaluate import evaluate
from figet.Constants import TYPE_VOCAB

log = get_logging()


class Coach(object):

    def __init__(self, model, optim, classifier, classifier_optim, vocabs, train_data, dev_data, test_data, hard_test_data, type2vec, args, extra_args):
        self.model = model
        self.model_optim = optim
        self.classifier = classifier
        self.classifier_optim = classifier_optim
        self.vocabs = vocabs
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.hard_test_data = hard_test_data
        self.args = args
        self.knn = kNN(vocabs[TYPE_VOCAB], type2vec, extra_args["knn_metric"] if "knn_metric" in extra_args else None)

    def train(self):
        log.debug(self.model)
        log.debug(self.classifier)

        self.start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)

            if epoch == self.args.epochs:
                log.info("\n\n------FINAL RESULTS----------")

            log.info("Validating on test data")
            test_loss, test_results = self.validate(self.test_data, epoch == self.args.epochs, epoch)
            test_eval = evaluate(test_results, verbose=True)
            log.info("Results epoch {}: Train loss: {:.2f}. Test loss: {:.2f}".format(epoch, train_loss, test_loss))
            log.info(test_eval)

        # log.info("HARD validation on HARD test data")
        # hard_test_results = self.validate(self.hard_test_data, show_positions=True)
        # log.info("HARD Results after {} epochs: Hard Test loss: {:.2f}".format(self.args.epochs, hard_test_results))

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_model_loss, total_classif_loss, total_avg_dist = [], [], []
        self.model.train()
        self.classifier.train()
        for i in tqdm(range(niter), desc="train_one_epoch"):
            batch = self.train_data[i]

            self.model_optim.zero_grad()

            model_loss, type_embeddings, _, avg_neg_dist, dist_to_pos, dist_to_neg = self.model(batch, epoch)
            model_loss.backward(retain_graph=True)
            self.model_optim.step()

            self.classifier_optim.zero_grad()
            _, classifier_loss = self.classifier(type_embeddings, batch[4])
            classifier_loss.backward()
            self.classifier_optim.step()

            # Stats.
            total_avg_dist.append(avg_neg_dist)
            total_model_loss.append(model_loss.item())
            total_classif_loss.append(classifier_loss.item())
            if (i + 1) % self.args.log_interval == 0:
                norms = torch.norm(type_embeddings, p=2, dim=1)
                mean_norm = norms.mean().item()
                max_norm = norms.max().item()
                min_norm = norms.min().item()
                avg_model_loss, avg_classif_loss = np.mean(total_model_loss), np.mean(total_classif_loss)

                hinge_neg_addition = len((dist_to_neg < avg_neg_dist * 0.6).nonzero())

                log.debug("Epoch %2d | %5d/%5d | loss %6.4f | %6.0f s elapsed"
                    % (epoch, i+1, len(self.train_data), avg_model_loss + avg_classif_loss, time.time()-self.start_time))
                log.debug(f"Model loss: {avg_model_loss}, Classif loss: {avg_classif_loss}")
                log.debug(f"Mean norm: {mean_norm:0.2f}, max norm: {max_norm}, min norm: {min_norm}")
                log.debug(f"avgs: d(true, neg): {np.mean(total_avg_dist)}, d to pos: {dist_to_pos.mean()}, d to neg: {dist_to_neg.mean()}, adding_to_loss:{hinge_neg_addition}/{len(dist_to_neg)}")

        return np.mean(total_model_loss) + np.mean(total_classif_loss)

    def validate(self, data, show_positions=False, epoch=None):
        total_model_loss, total_classif_loss = [], []
        results = []
        true_positions = []
        k = 15
        among_top_k, total = 0, 0
        self.model.eval()
        self.classifier.eval()
        log_interval = len(data) / 2
        for i in range(len(data)):
            batch = data[i]
            types = batch[3]
            one_hot_types = batch[4]

            model_loss, type_embeddings, _, _, _, _ = self.model(batch, epoch)
            predictions, classifier_loss = self.classifier(type_embeddings, one_hot_types)

            total_model_loss.append(model_loss.item())
            total_classif_loss.append(classifier_loss.item())

            results += assign_types(predictions, types)

            among_top_k += self.knn.precision_at(type_embeddings, types, k=k)
            total += len(types)

            if show_positions:
                true_positions.extend(self.knn.true_types_position(type_embeddings, types))

            if i % log_interval == 0:
                log.debug("Processing batch {} of {}".format(i, len(data)))

        if show_positions:
            log.info("Positions: Mean:{:.2f} Std: {:.2f}".format(np.mean(true_positions), np.std(true_positions)))
            proportion = sum(val < k for val in true_positions) / float(len(true_positions)) * 100
            log.info("Proportion of neighbors in first {}: {}".format(k, proportion))
            proportion = sum(val < 2 * k for val in true_positions) / float(len(true_positions)) * 100
            log.info("Proportion of neighbors in first {}: {}".format(2*k, proportion))

        log.info("Precision@{}: {:.2f}".format(k, float(among_top_k) * 100 / total))

        return np.mean(total_model_loss) + np.mean(total_classif_loss), results
