#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import random
from torch import nn
from torch.optim import Adam

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
parser.add_argument("--context_length", default=10, type=int, help="Max length of the left/right context.")
# parser.add_argument("--context_input_size", default=300, type=int, help="Input size of ContextEncoder.")
parser.add_argument("--context_rnn_size", default=200, type=int, help="RNN size of ContextEncoder.")
parser.add_argument("--context_num_layers", default=1, type=int, help="Number of layers of ContextEncoder.")
parser.add_argument("--context_num_directions", default=2, choices=[1, 2], type=int,
                    help="Number of directions for ContextEncoder RNN.")
parser.add_argument("--attn_size", default=100, type=int, help="Attention vector size.")
parser.add_argument("--single_context", default=0, type=int, help="Use single context.")
parser.add_argument("--negative_samples", default=10, type=int, help="Amount of negative samples.")
parser.add_argument("--neighbors", default=30, type=int, help="Amount of neighbors to analize.")

# Other parameters
parser.add_argument("--bias", default=0, type=int, help="Whether to use bias in the linear transformation.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Starting learning rate.")
parser.add_argument("--l2", default=0.00, type=float, help="L2 Regularization.")
parser.add_argument("--param_init", default=0.01, type=float,
                    help=("Parameters are initialized over uniform distribution"
                          "with support (-param_init, param_init)"))
parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate for all applicable modules.")
parser.add_argument("--niter", default=150, type=int, help="Number of iterations per epoch.")
parser.add_argument("--epochs", default=15, type=int, help="Number of training epochs.")
parser.add_argument("--max_grad_norm", default=-1, type=float,
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
    dataset.create_one_hot_types()
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
    hard_test_data = get_dataset(data, args, "hard_test")
    negative_samples = data["negative_samples"]

    log.debug("Loading word2vecs from '%s'." % args.word2vec)
    word2vec = torch.load(args.word2vec)
    log.debug("Loading type2vecs from '%s'." % args.type2vec)
    type2vec = torch.load(args.type2vec)

    args.context_input_size = word2vec.size()[1]
    args.type_dims = type2vec.size()[1]

    proj_learning_rate = [0.01]      # This doesn't affect at all
    proj_weight_decay = [0.0]        # This doesn't affect at all
    proj_bias = [0]                  # This doesn't affect at all
    proj_non_linearity = [None]      # This doesn't affect at all

    classif_learning_rate = [0.001, 0.0005, 0.0002]
    classif_weight_decay = [0.001]
    classif_bias = [0, 1]
    classif_dropout = [0.25, 0.5]
    classif_hidden_size = [500, 700]

    neighbors = [3]
    knn_metrics = [hyperbolic_distance_numpy]

    configs = itertools.product(proj_learning_rate, proj_weight_decay, proj_bias, proj_non_linearity,
                                classif_learning_rate, classif_weight_decay, classif_bias, classif_dropout, classif_hidden_size,
                                neighbors, knn_metrics)

    best_strict, best_macro, best_micro = -1,-1, -1
    best_strict_result, best_macro_result, best_micro_result = None, None, None
    best_strict_config, best_macro_config, best_micro_config = None, None, None

    for config in configs:

        extra_args = {"knn_metric": config[10], "activation_function": config[3]}

        args.proj_learning_rate = config[0]
        args.proj_weight_decay = config[1]
        args.proj_bias = config[2]

        args.classif_bias = config[6]
        args.classif_dropout = config[7]
        args.classif_hidden_size = config[8]

        args.neighbors = config[9]

        log.debug("Building model...")
        model = figet.Models.Model(args, vocabs, negative_samples, extra_args)
        classifier = figet.Classifier(args, vocabs, type2vec)
        classifier_optim = Adam(classifier.parameters(), lr=config[4], weight_decay=config[5])

        if len(args.gpus) >= 1:
            model.cuda()
            classifier.cuda()

        log.debug("Copying embeddings to model...")
        model.init_params(word2vec, type2vec)
        optim = figet.Optim(model.parameters(), args.proj_learning_rate, args.max_grad_norm, args.proj_weight_decay)

        nParams = sum([p.nelement() for p in model.parameters()]) + sum([p.nelement() for p in classifier.parameters()])
        log.debug("* number of parameters: %d" % nParams)

        coach = figet.Coach(model, optim, classifier, classifier_optim, vocabs, train_data, dev_data, test_data, hard_test_data, type2vec, hierarchy, args, extra_args)

        # Train.
        log.info("Start training...")
        log_config(config)
        results = coach.train()
        log.info("Done!\n\n")

        strict_f1, macro_f1, micro_f1 = results[0][-1], results[1][-1], results[2][-1]

        if strict_f1 > best_strict:
            best_strict = strict_f1
            best_strict_config = config[:]
            best_strict_result = results
            log.info("Best strict found!!!")
            log.info(results)
            log_config(config)

        if macro_f1 > best_macro:
            best_macro = macro_f1
            best_macro_config = config[:]
            best_macro_result = results
            log.info("Best macro found!!!")
            log.info(results)
            log_config(config)

        if micro_f1 > best_micro:
            best_micro = micro_f1
            best_micro_config = config[:]
            best_micro_result = results
            log.info("Best micro found!!!")
            log.info(results)
            log_config(config)

    log.info("\n\n-----FINAL FINAL----------")
    log.info("BEST STRICT CONFIG")
    log.info(best_strict_result)
    log_config(best_strict_config)
    log.info("BEST MACRO CONFIG")
    log.info(best_macro_result)
    log_config(best_macro_config)
    log.info("BEST MICRO CONFIG")
    log.info(best_micro_result)
    log_config(best_micro_config)


def log_config(config):
    log.info(f"proj_lr:{config[0]}, proj_l2:{config[1]}, proj_bias:{config[2]}, proj_nonlin:{config[3]}, "
             f"classif_lr:{config[4]}, cl_l2:{config[5]}, cl_bias:{config[6]}, cl_dropout:{config[7]}, cl_hidden:{config[8]}, "
             f"Neighbors:{config[9]}, knn:{config[10]}")


if __name__ == "__main__":
    main()
