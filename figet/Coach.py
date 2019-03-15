#!/usr/bin/env python
# encoding: utf-8

import copy
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from statistics import mean, stdev, median, mode, StatisticsError
from tensorboardX import SummaryWriter

from figet.utils import get_logging, plot_k
from figet.Predictor import kNN, assign_types, assign_all_granularities_types
from figet.evaluate import evaluate, raw_evaluate, stratified_evaluate
from figet.Constants import TYPE_VOCAB, COARSE_FLAG, FINE_FLAG, UF_FLAG
from figet.result_printer import ResultPrinter

log = get_logging()


class Coach(object):

    def __init__(self, model, optim, vocabs, train_data, dev_data, test_data, hard_test_data, type2vec, word2vec, hierarchy, args, extra_args, config):
        self.model = model
        self.model_optim = optim
        self.vocabs = vocabs
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.hard_test_data = hard_test_data
        self.hierarchy = hierarchy
        self.args = args
        self.word2vec = word2vec
        self.type2vec = type2vec
        self.knn = kNN(type2vec, vocabs[TYPE_VOCAB])
        self.result_printer = ResultPrinter(dev_data, vocabs, model, None, self.knn, hierarchy, args)
        self.config = config
        self.granularities = [COARSE_FLAG, FINE_FLAG, UF_FLAG]
        self.writer = SummaryWriter(f"board/{args.exp_name}")

    def train(self):
        log.debug(self.model)

        max_coarse_macro_f1, best_model_state, best_epoch = -1, None, 0

        for epoch in range(1, self.args.epochs + 1):
            train_model_loss = self.train_epoch(epoch)
            self.writer.add_scalar("train/epoch_loss", train_model_loss, epoch)

            log.info(f"Results epoch {epoch}: TRAIN loss: model: {train_model_loss:.2f}")

            results, _ = self.validate_typing(self.dev_data, "dev", epoch)
            _, coarse_results = stratified_evaluate(results[0], self.vocabs[TYPE_VOCAB])
            coarse_split = coarse_results.split()
            coarse_macro_f1 = float(coarse_split[5])

            self.writer.add_scalar("dev_strict_f1", float(coarse_split[2]), epoch)
            self.writer.add_scalar("dev_macro_f1", coarse_macro_f1, epoch)
            self.writer.add_scalar("dev_micro_f1", float(coarse_split[8]), epoch)

            if coarse_macro_f1 > max_coarse_macro_f1:
                max_coarse_macro_f1 = coarse_macro_f1
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                log.info(f"* Best coarse macro F1 {coarse_macro_f1:0.2f} at epoch {epoch} *")

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            if epoch % 30 == 0:
                self.print_full_validation(self.dev_data, f"epoch-{epoch}-dev")
                torch.save(self.model.state_dict(), f"models/sep-grad-corrected-lr1p0-dict-{epoch}.pt")

        log.info(f"Final evaluation on best coarse macro F1 ({max_coarse_macro_f1}) from epoch {best_epoch}")
        self.model.load_state_dict(best_model_state)

        self.result_printer.show()

        # self.validate_all_neighbors(self.dev_data, "dev", plot=True)
        self.print_full_validation(self.dev_data, "dev")

        # self.validate_all_neighbors(self.test_data, "test", plot=True)
        coarse_results = self.print_full_validation(self.test_data, "test")

        self.writer.close()

        return coarse_results

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        if self.args.extra_shuffle == 1:
            self.train_data.shuffle()

        niter = self.args.niter if self.args.niter != -1 else len(self.train_data)  # -1 in train and len(self.train_data) is num_batches
        total_model_loss = []
        # angles, dist_to_pos, euclid_dist, norms
        stats = [[[], [], [], []],
                 [[], [], [], []],
                 [[], [], [], []]]

        self.set_learning_rate(epoch)
        self.model.train()
        for i in tqdm(range(niter), desc="train_epoch_{}".format(epoch)):
            batch = self.train_data[i]

            self.model_optim.zero_grad()
            model_loss, type_embeddings, _, _, angles, dist_to_pos, euclid_dist = self.model(batch, epoch)
            model_loss.backward()
            self.write_norm(epoch, i)
            if self.args.max_grad_norm >= 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.model_optim.step()

            # Stats.
            for idx, item in enumerate(stats):
                item[0].append(angles[idx].mean().item())
                item[1].append(dist_to_pos[idx].mean().item())
                item[2].append(euclid_dist[idx].mean().item())
                item[3].append(torch.norm(type_embeddings[idx].detach(), p=2, dim=1).mean().item())

            total_model_loss.append(model_loss.item())

        self.print_stats(stats, "train", epoch)
        return np.mean(total_model_loss)

    def print_full_validation(self, dataset, name):
        log.info(f"\n\n\nVALIDATION ON {name.upper()}")
        gran_true_and_pred, total_true_and_pred = self.validate_typing(dataset, name, -1)
        coarse_results = None
        for title, set_true_and_pred in zip(["COARSE", "FINE", "ULTRAFINE", "TOTAL"], gran_true_and_pred + [total_true_and_pred]):
            combined_eval = evaluate(set_true_and_pred)
            stratified_eval, coarse_eval = stratified_evaluate(set_true_and_pred, self.vocabs[TYPE_VOCAB])
            if title == "COARSE":
                coarse_results = coarse_eval
            log.info(f"\nRESULTS ON {title}")
            log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)\n" + combined_eval)
            log.info(f"Final Stratified evaluation on {name.upper()}:\n" + stratified_eval)
        return coarse_results

    def validate_typing(self, data, name, epoch):
        total_model_loss = []
        # angles, dist_to_pos, euclid_dist, norms
        stats = [[[], [], [], []],
                 [[], [], [], []],
                 [[], [], [], []]]
        results = [[], [], []]
        total_result = []
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc=f"validate_typing_{name}_{epoch}"):
                batch = data[i]
                types = batch[5]

                model_loss, predicted_embeds, feature_repre, _, angles, dist_to_pos, euclid_dist = self.model(batch, 0)

                neighbor_indexes = []
                for gran_flag, pred in zip(self.granularities, predicted_embeds):
                    neighbor_indexes.append(self.knn.neighbors(pred, -1, gran_flag))

                for gran_flag, (idx, neighs) in zip(self.granularities, enumerate(neighbor_indexes)):
                    results[idx] += assign_types(predicted_embeds[gran_flag], neighs, types, self.knn, gran_flag)
                total_result += assign_all_granularities_types(predicted_embeds, neighbor_indexes, types, self.knn)

                # collect stats
                total_model_loss.append(model_loss.item())
                for idx, item in enumerate(stats):
                    item[0].append(angles[idx].mean().item())
                    item[1].append(dist_to_pos[idx].mean().item())
                    item[2].append(euclid_dist[idx].mean().item())
                    item[3].append(torch.norm(predicted_embeds[idx].detach(), p=2, dim=1).mean().item())

            self.print_stats(stats, name, epoch)
            log.debug(f"{name} loss: {np.mean(total_model_loss)}")
            if epoch != -1:
                self.writer.add_scalar(f"{name}/epoch_loss", np.mean(total_model_loss), epoch)

            return results, total_result

    def print_stats(self, stats, name, epoch):
        labels = ["COARSE", "FINE", "ULTRAFINE"]
        for idx, item in enumerate(stats):
            log.debug(
                f"\nProj {name} epoch {epoch} {labels[idx]}: d to pos: {mean(item[1]):0.2f} +- {stdev(item[1]):0.2f}, "
                f"Euclid: {mean(item[2]):0.2f} +- {stdev(item[2]):0.2f}, "
                f"Angles: {mean(item[0]):0.2f} +- {stdev(item[0]):0.2f}, "
                f"Mean norm:{mean(item[3]):0.2f}+-{stdev(item[3]):0.2f}")

            if epoch != -1:
                prefix = f"Proj/{name}_{labels[idx]}"
                self.writer.add_scalar(f"{prefix}_d_to_pos", mean(item[1]), epoch)
                self.writer.add_scalar(f"{prefix}_euclid", mean(item[2]), epoch)
                self.writer.add_scalar(f"{prefix}_angles", mean(item[0]), epoch)
                self.writer.add_scalar(f"{prefix}_norm", mean(item[3]), epoch)

    def validate_all_neighbors(self, data, name, plot=False):
        """Warning: this function is very slow because it evaluates over all types on the dataset"""
        # full, closest
        positions = [[[], []],
                     [[], []],
                     [[], []]]

        log.info(f"Validating projection on {name.upper()} data")
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc=f"validate_proj_{name}"):
                batch = data[i]
                types = batch[5]

                model_loss, predicted_embeds, feature_repre, _, angles, dist_to_pos, euclid_dist = self.model(batch, 0)

                for gran_flag, (idx, item) in zip(self.granularities, enumerate(positions)):
                    type_positions, closest_true_neighbor = self.knn.type_positions(predicted_embeds[gran_flag], types, gran_flag)
                    item[0].extend(type_positions)
                    item[1].extend(closest_true_neighbor)

            labels = ["COARSE", "FINE", "ULTRAFINE"]
            for idx, item in enumerate(positions):
                self.log_neighbor_positions(item[1], f"{labels[idx]} CLOSEST", self.args.neighbors)
                self.log_neighbor_positions(item[0], f"{labels[idx]} FULL", self.args.neighbors)

            if plot:
                plot_k(f"{name}_COARSE", positions[0][0], positions[0][1])
                plot_k(f"{name}_FINE", positions[1][0], positions[1][1])
                plot_k(f"{name}_UF", positions[2][0], positions[2][1])

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

    def write_norm(self, epoch, iter_i):
        t = ["coarse", "fine", "ultrafine"]
        for granularity, layer in zip(t, [self.model.coarse_projector.W_out, self.model.fine_projector.W_out, self.model.ultrafine_projector.W_out]):
            gradient = layer.weight.grad
            if gradient is not None:
                grad_norm = gradient.data.norm(2).item()
                self.writer.add_scalar(f"norm_{granularity}/epoch_{epoch}", grad_norm, iter_i)

