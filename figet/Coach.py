#!/usr/bin/env python
# encoding: utf-8

import time
import copy
import numpy as np
from tqdm import tqdm
import torch

import figet
from figet.Predictor import Predictor
from figet.Constants import TYPE_VOCAB

log = figet.utils.get_logging()


class Coach(object):

    def __init__(self, model, vocabs, train_data, dev_data, test_data, optim, type2vec, args):
        self.model = model
        self.vocabs = vocabs
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.optim = optim
        self.args = args
        self.predictor = Predictor(vocabs[TYPE_VOCAB], type2vec)

    def train(self):
        log.debug(self.model)

        start_train_time = time.time()
        self.start_time = time.time()
        # Early stopping.
        best_dev_f1, best_epoch, best_state = None, None, None
        # Adaptive threshods.
        best_dev_dist, dev_labels = None, None
        dev_results, train_loss = None, None

        validation_steps = 5

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)

            log.info("Validating on test data")
            test_results = self.validate(self.test_data, epoch == self.args.epochs)
            log.info("Results epoch {}: Train loss: {:.2f}. Test loss: {:.2f}".format(epoch, train_loss * 100, test_results))

            # if epoch % validation_steps == 0:
            #     log.info("Validating on dev data")
            #     dev_results = self.validate(self.dev_data)
            #     dev_acc = figet.evaluate.evaluate(dev_results[1])
            #
            #     log.info("Epoch {} | Dev strict acc. {} | Loss train {:.2f} | Loss dev {:.2f} |".format(
            #              epoch, dev_acc, train_loss * 100, dev_results[0] * 100))
            #
            #     dev_f1_strict = figet.evaluate.strict(dev_results[1])[2]
            #     if best_dev_f1 is None or dev_f1_strict > best_dev_f1:
            #         best_dev_f1 = dev_f1_strict
            #         best_epoch = epoch
            #         best_state = copy.deepcopy(self.model.state_dict())
            #         best_dev_dist, dev_labels = dev_results[2:4]
            #         log.info("NEW best dev at epoch {} F1: {:.2f}".format(epoch, best_dev_f1 * 100))

        # log.info("Best Dev F1: {:.2f} at epoch {}".format(best_dev_f1 * 100, best_epoch))
        #
        # log.info("Validating on train data")
        # train_results = self.validate(self.train_data.subsample(self.test_data.batch_size * self.test_data.num_batches))
        # log.info("Train loss: {:.2f}".format(train_results[0]))
        #
        # log.info("Validating on test data")
        # test_results = self.validate(self.test_data)
        # log.info("Test loss: {:.2f}".format(test_results[0]))

        #
        # log.info("FINAL results: train acc, dev acc, test acc, loss (tr,d,te)")
        # # log.info("\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}".format(
        # #     train_acc, dev_acc, test_acc, train_loss * 100, dev_results[0] * 100, test_results[0] * 100))
        # minutes = int((time.time() - start_train_time) / 60)
        # log.info("Total training time (min): {}, Epochs: {}, avg epoch time: {} mins".format(minutes, self.args.epochs, minutes / self.args.epochs))

        # return best_state, best_dev_dist, dev_labels, test_results[2], test_results[3]

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_loss, report_loss = [], []
        self.model.train()
        for i in tqdm(range(niter), desc="train_one_epoch"):
            batch = self.train_data[i]

            self.model.zero_grad()
            loss, _, _ = self.model(batch)
            self.optim.optimizer.zero_grad()
            loss.backward()
            self.optim.step()

            # Stats.
            total_loss += [loss.item()]
            report_loss += [loss.item()]
            if (i + 1) % self.args.log_interval == 0:
                log.debug("Epoch %2d | %5d/%5d | loss %6.2f | %6.0f s elapsed"
                    % (epoch, i+1, len(self.train_data), np.mean(report_loss), time.time()-self.start_time))

        return np.mean(total_loss)

    def validate(self, data, show_positions=False):
        total_loss = []
        true_positions = []
        k = 20
        among_top_k, total = 0, 0
        self.model.eval()
        log_interval = len(data) / 4
        for i in range(len(data)):
            batch = data[i]
            types = batch[3]    # es un indice que indica que type es
            loss, dist, _ = self.model(batch)   # dist es el vector predicho
            total_loss.append(loss.item())

            among_top_k += self.predictor.precision_at(dist.data, types.data, k=k)
            total += len(types)

            true_positions.extend(self.predictor.true_types_position(dist.data, types.data))

            if i % log_interval == 0:
                log.debug("Processing batch {} of {}".format(i, len(data)))

        if show_positions:
            log.info("\n\n{}\n\n".format(true_positions))
            np_pos_array = np.array(true_positions)
            torch_pos_array = torch.FloatTensor(true_positions)
            log.info("\nnumpy: mean:{:.2f} std: {:.2f}\ntorch: mean:{:.2f} std: {:.2f}".format(
                np_pos_array.mean(),np_pos_array.std(), torch_pos_array.mean(), torch_pos_array.std()))
        log.info("Precision@{}: {:.2f}".format(k, float(among_top_k) * 100 / total))
        return np.mean(total_loss)


# total_loss = []
        # self.model.eval()
        # predictions = []
        # dists, labels = [], []
        # log_interval = len(data) / 4
        # for i in range(len(data)):
        #     batch = data[i]
        #     types = batch[3]
        #     loss, dist, attn = self.model(batch)
        #
        #     predictions.extend(figet.adaptive_thres.predict(dist.data, types.data))
        #     dists.append(dist.data)
        #     labels.append(types.data)
        #     total_loss.append(loss.item())
        #
        #     if i % log_interval == 0:
        #         log.debug("Processing batch {} of {}".format(i, len(data)))
        #
        # dists = torch.cat(dists, 0)
        # labels = torch.cat(labels, 0)
        # return np.mean(total_loss), predictions, dists.cpu().numpy(), labels.cpu().numpy()
