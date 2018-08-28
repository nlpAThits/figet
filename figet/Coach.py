#!/usr/bin/env python
# encoding: utf-8

import time
import torch
import numpy as np
from tqdm import tqdm

from figet.utils import get_logging
from figet.Predictor import Predictor
from figet.Constants import TYPE_VOCAB

log = get_logging()


class Coach(object):

    def __init__(self, model, vocabs, train_data, dev_data, test_data, hard_test_data, optim, type2vec, args, extra_args):
        self.model = model
        self.vocabs = vocabs
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.hard_test_data = hard_test_data
        self.optim = optim
        self.args = args
        self.predictor = Predictor(vocabs[TYPE_VOCAB], type2vec, extra_args["knn_metric"] if "knn_metric" in extra_args else None)

    def train(self):
        log.debug(self.model)

        self.start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)

            if epoch == self.args.epochs:
                log.info("\n\n------FINAL RESULTS----------")

            log.info("Validating on test data")
            test_results = self.validate(self.test_data, epoch == self.args.epochs, epoch)
            log.info("Results epoch {}: Train loss: {:.2f}. Test loss: {:.2f}".format(epoch, train_loss, test_results))

        # log.info("HARD validation on HARD test data")
        # hard_test_results = self.validate(self.hard_test_data, show_positions=True)
        # log.info("HARD Results after {} epochs: Hard Test loss: {:.2f}".format(self.args.epochs, hard_test_results))

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_loss, report_loss, total_avg_dist = [], [], []
        self.model.train()
        for i in tqdm(range(niter), desc="train_one_epoch"):
            batch = self.train_data[i]

            self.optim.zero_grad()
            loss, predictions, _, avg_neg_dist, dist_to_pos, dist_to_neg = self.model(batch, epoch)

            loss.backward()

            self.optim.step()

            # Stats.
            total_avg_dist.append(avg_neg_dist)
            total_loss.append(loss.item())
            report_loss.append(loss.item())
            if (i + 1) % self.args.log_interval == 0:
                norms = torch.norm(predictions, p=2, dim=1)
                mean_norm = norms.mean().item()
                max_norm = norms.max().item()
                min_norm = norms.min().item()

                hinge_neg_addition = len((dist_to_neg < avg_neg_dist * 0.6).nonzero())

                log.debug("Epoch %2d | %5d/%5d | loss %6.4f | %6.0f s elapsed"
                    % (epoch, i+1, len(self.train_data), np.mean(report_loss), time.time()-self.start_time))
                log.debug(f"Mean norm: {mean_norm:0.2f}, max norm: {max_norm}, min norm: {min_norm}")
                log.debug(f"avgs: d(true, neg): {np.mean(total_avg_dist)}, d to pos: {dist_to_pos.mean()}, d to neg: {dist_to_neg.mean()}, adding_to_loss:{hinge_neg_addition}/{len(dist_to_neg)}")

        return np.mean(total_loss)

    def validate(self, data, show_positions=False, epoch=None):
        total_loss = []
        true_positions = []
        k = 100
        among_top_k, total = 0, 0
        self.model.eval()
        log_interval = len(data) / 4
        for i in range(len(data)):
            batch = data[i]
            types = batch[3]
            loss, dist, _, _, _, _ = self.model(batch, epoch)
            total_loss.append(loss.item())

            among_top_k += self.predictor.precision_at(dist.data, types.data, k=k)
            total += len(types)

            if show_positions:
                true_positions.extend(self.predictor.true_types_position(dist.data, types.data))

            # if i % log_interval == 0:
                # log.debug("Processing batch {} of {}".format(i, len(data)))

        if show_positions:
            log.info("Positions: Mean:{:.2f} Std: {:.2f}".format(np.mean(true_positions), np.std(true_positions)))
            proportion = sum(val < 100 for val in true_positions) / float(len(true_positions)) * 100
            log.info("Proportion of neighbors in first 100: {}".format(proportion))
            proportion = sum(val < 200 for val in true_positions) / float(len(true_positions)) * 100
            log.info("Proportion of neighbors in first 200: {}".format(proportion))

        log.info("Precision@{}: {:.2f}".format(k, float(among_top_k) * 100 / total))
        return np.mean(total_loss)
