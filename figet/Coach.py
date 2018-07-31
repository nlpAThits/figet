#!/usr/bin/env python
# encoding: utf-8

import time
import copy
import numpy as np
from tqdm import tqdm
import torch

import figet

log = figet.utils.get_logging()


class Coach(object):

    def __init__(self, model, vocabs, train_data, dev_data, test_data, optim, args):
        self.model = model
        self.vocabs = vocabs
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.optim = optim
        self.args = args

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

            if epoch % validation_steps == 0:

                dev_results = self.validate(self.dev_data)
                dev_acc = figet.evaluate.evaluate(dev_results[1])

                log.info("Epoch {} | Dev strict acc. {} | Loss train {:.2f} | Loss dev {:.2f} |".format(
                         epoch, dev_acc, train_loss * 100, dev_results[0] * 100))

                dev_f1_strict = figet.evaluate.strict(dev_results[1])[2]
                if best_dev_f1 is None or dev_f1_strict > best_dev_f1:
                    best_dev_f1 = dev_f1_strict
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_dev_dist, dev_labels = dev_results[2:4]
                    log.info("NEW best dev at epoch {} F1: {:.2f}".format(epoch, best_dev_f1))

        log.info("Best Dev F1: {:.2f} at epoch {}".format(best_dev_f1, best_epoch))

        test_results = self.validate(self.test_data)
        test_acc = figet.evaluate.evaluate(test_results[1])

        log.info("FINAL results: dev acc, test acc, loss (tr,d,te)")
        log.info("\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            dev_acc, test_acc, train_loss * 100, dev_results[0] * 100, test_results[0] * 100))
        log.info("Total training time (minutes): {}".format(int((time.time() - start_train_time) / 60)))

        return best_state, best_dev_dist, dev_labels, test_results[2], test_results[3]

    def log_results(self, data, name, logging=False):
        results = self.validate(data)

        if logging:
            log.info("Validating on {} data".format(name))
            log.info("Strict, Macro, Micro, loss ")
            log.info("%s\t%.2f\t%.2f\t%.2f".format(figet.evaluate.evaluate(results[1]), results[0] * 100))

        prob_dist, labels = results[2], results[3]
        return prob_dist, labels

    def train_epoch(self, epoch):
        """
        :param epoch: int >= 1
        """
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_tokens, report_tokens = 0, 0
        total_loss, report_loss = [], []
        start_time = time.time()
        self.model.train()
        for i in tqdm(range(niter), desc="train_one_epoch"):
            batch = self.train_data[i]

            self.model.zero_grad()
            loss, _, _ = self.model(batch)
            self.optim.optimizer.zero_grad()
            loss.backward()
            self.optim.step()

            # Stats.            # Creo que la mayoria de esto no se muestra nunca!!!
            prev_context, next_context = batch[1], batch[2]
            if isinstance(prev_context, tuple):
                num_tokens = prev_context[0][0].data.ne(figet.Constants.PAD).sum()
            else:
                num_tokens = prev_context.data.ne(figet.Constants.PAD).sum()
            if self.args.single_context == 0:
                if isinstance(next_context, tuple):
                    num_tokens = next_context[0][0].data.ne(figet.Constants.PAD).sum()
                else:
                    num_tokens = next_context.data.ne(figet.Constants.PAD).sum()
            total_tokens += num_tokens
            report_tokens += num_tokens
            total_loss += [loss.item()]
            report_loss += [loss.item()]
            if i % self.args.log_interval == -1 % self.args.log_interval:
                log.debug(
                    "Epoch %2d | %5d/%5d | loss %6.2f | %3.0f ctx tok/s | %6.0f s elapsed"
                    %(epoch, i+1, len(self.train_data), np.mean(report_loss),
                      report_tokens/(time.time()-start_time),
                      time.time()-self.start_time))
                report_tokens = 0
                report_loss = []
                start_time = time.time()

        return np.mean(total_loss)

    def validate(self, data):
        total_loss = []
        self.model.eval()
        predictions = []
        dists, labels = [], []
        for i in range(len(data)):
            batch = data[i]
            types = batch[3]
            loss, dist, attn = self.model(batch)

            predictions.extend(figet.adaptive_thres.predict(dist.data, types.data))
            dists.append(dist.data)
            labels.append(types.data)
            total_loss.append(loss.item())

        dists = torch.cat(dists, 0)
        labels = torch.cat(labels, 0)
        return np.mean(total_loss), predictions, dists.cpu().numpy(), labels.cpu().numpy()
