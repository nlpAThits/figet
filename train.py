#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import random
from torch import nn
from torch.optim import SGD, Adam

import figet
from figet.hyperbolic import *
import itertools


parser = argparse.ArgumentParser("train.py")

# Data options
parser.add_argument("--data", required=True, type=str, help="Data path.")
parser.add_argument("--save_tuning", default="./save/tuning.pt", type=str,
                    help="Save the intermediate results for tuning.")
parser.add_argument("--save_model", default="./save/model.pt", type=str, help="Save the model.")

# Sentence-level context parameters
parser.add_argument("--emb_size", default=300, type=int, help="Embedding size.")
parser.add_argument("--char_emb_size", default=50, type=int, help="Char embedding size.")
parser.add_argument("--positional_emb_size", default=25, type=int, help="Positional embedding size.")
parser.add_argument("--context_rnn_size", default=200, type=int, help="RNN size of ContextEncoder.")

parser.add_argument("--attn_size", default=100, type=int, help="Attention vector size.")
parser.add_argument("--negative_samples", default=10, type=int, help="Amount of negative samples.")
parser.add_argument("--neighbors", default=30, type=int, help="Amount of neighbors to analize.")

# Other parameters
parser.add_argument("--bias", default=0, type=int, help="Whether to use bias in the linear transformation.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Starting learning rate.")
parser.add_argument("--l2", default=0.00, type=float, help="L2 Regularization.")
parser.add_argument("--param_init", default=0.01, type=float,
                    help=("Parameters are initialized over uniform distribution"
                          "with support (-param_init, param_init)"))
parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
parser.add_argument("--mention_dropout", default=0.5, type=float, help="Dropout rate for mention")
parser.add_argument("--context_dropout", default=0.2, type=float, help="Dropout rate for context")
parser.add_argument("--niter", default=150, type=int, help="Number of iterations per epoch.")
parser.add_argument("--epochs", default=15, type=int, help="Number of training epochs.")
parser.add_argument("--max_grad_norm", default=5, type=float,
                    help="""If the norm of the gradient vector exceeds this, 
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument("--extra_shuffle", default=1, type=int,
                    help="""By default only shuffle mini-batch order; when true, shuffle and re-assign mini-batches""")
parser.add_argument('--seed', type=int, default=3435, help="Random seed")
parser.add_argument("--word2vec", default=None, type=str, help="Pretrained word vectors.")
parser.add_argument("--type2vec", default=None, type=str, help="Pretrained type vectors.")
parser.add_argument("--gpus", default=[], nargs="+", type=int, help="Use CUDA on the listed devices.")
parser.add_argument('--log_interval', type=int, default=1000, help="Print stats at this interval.")

args = parser.parse_args()

if args.gpus:
    torch.cuda.set_device(args.gpus[0])

seed = random.randint(1, 100000)
figet.utils.set_seed(seed)

log = figet.utils.get_logging()
log.debug(args)


def get_dataset(data, args, key):
    dataset = data[key]
    dataset.set_batch_size(args.batch_size)
    return dataset


def main():
    # Load data.
    log.debug("Loading data from '%s'." % args.data)
    data = torch.load(args.data)
    vocabs = data["vocabs"]
    hierarchy = data["hierarchy"]

    # datasets
    train_data = get_dataset(data, args, "train")
    dev_data = get_dataset(data, args, "dev")
    test_data = get_dataset(data, args, "test")
    # hard_test_data = get_dataset(data, args, "hard_test")
    # negative_samples = data["negative_samples"]

    log.debug("Loading word2vecs from '%s'." % args.word2vec)
    word2vec = torch.load(args.word2vec)
    log.debug("Loading type2vecs from '%s'." % args.type2vec)
    type2vec = torch.load(args.type2vec)

    args.type_dims = type2vec.size(1)

    proj_learning_rate = [0.05]
    proj_weight_decay = [0.0]
    proj_bias = [1]
    proj_hidden_layers = [1]
    proj_hidden_size = [150]
    proj_non_linearity = [None]         # not used
    proj_dropout = [0.3]

    k_neighbors = [4]
    args.knn_hyper = True

    cosine_factors = [50]
    hyperdist_factors = [1]

    configs = itertools.product(proj_learning_rate, proj_weight_decay, proj_bias, proj_non_linearity, proj_dropout,
                                proj_hidden_layers, proj_hidden_size, cosine_factors, hyperdist_factors, k_neighbors)

    best_coarse_macro_f1 = -1
    best_configs, best_coarse_results = [], []

    for config in configs:

        extra_args = {"activation_function": config[3]}

        args.proj_learning_rate = config[0]
        args.proj_weight_decay = config[1]
        args.proj_bias = config[2]
        args.proj_dropout = config[4]
        args.proj_hidden_layers = config[5]
        args.proj_hidden_size = config[6]

        args.cosine_factor = config[7]
        args.hyperdist_factor = config[8]

        args.neighbors = config[9]

        log.debug("Building model...")
        model = figet.Models.Model(args, vocabs, None, extra_args)

        if len(args.gpus) >= 1:
            model.cuda()

        log.debug("Copying embeddings to model...")
        model.init_params(word2vec, type2vec)
        optim = SGD(model.parameters(), lr=args.proj_learning_rate, weight_decay=args.proj_weight_decay)

        nParams = sum([p.nelement() for p in model.parameters()])
        log.debug("* number of parameters: %d" % nParams)

        coach = figet.Coach(model, optim, vocabs, train_data, dev_data, test_data, None, type2vec, word2vec, hierarchy, args, extra_args, config)

        # Train.
        log.info("Start training...")
        log_config(config)
        coarse_result_string = coach.train()
        coarse_macro_f1 = float(coarse_result_string.split()[5])

        if coarse_macro_f1 > best_coarse_macro_f1:
            best_coarse_macro_f1 = coarse_macro_f1
            best_configs.append(config[:])
            best_coarse_results.append(coarse_result_string)

        log_config(config)
        log.info("Done!\n\n")

    log.info("3rd best result")
    print_final_results(best_configs, best_coarse_results, -3)
    log.info("\n\n2nd best result")
    print_final_results(best_configs, best_coarse_results, -2)
    log.info("\n\nBEST RESULT")
    print_final_results(best_configs, best_coarse_results, -1)


def log_config(config):
    log.info(f"proj_lr:{config[0]}, proj_l2:{config[1]}, proj_bias:{config[2]}, proj_nonlin:{config[3]}, "
             f"proj_dropout:{config[4]}, proj_hidden_layers: {config[5]}, proj_hidden_size:{config[6]}, "
             f"cosine_factor:{config[7]}, hyperdist_factor:{config[8]}, neighbors: {config[8]}")


def print_final_results(best_configs, best_coarse_results, index):
    try:
        log.info(f"Test eval over coarse:\n{best_coarse_results[index]}")
        log.info(f"Config")
        log_config(best_configs[index])
    except IndexError:
        return


if __name__ == "__main__":
    main()
