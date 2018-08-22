#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import random
import torch.nn as nn

import figet
from figet.Loss import *


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
    return dataset


def main():
    # Load data.
    log.debug("Loading data from '%s'." % args.data)
    data = torch.load(args.data)
    vocabs = data["vocabs"]

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

    weight_decay = [0.0, 0.001, 0.01]
    learning_rate = [0.001, 0.01]
    # loss_metrics = [PoincareDistance.apply, nn.PairwiseDistance(p=2)]

    knn_metrics = [hyperbolic_distance_numpy, None]
    # weight_decay = [0.0]
    # learning_rate = [0.01]
    bias = [0, 1]
    non_linearity = [None, nn.Tanh()]

    for knn_metric in knn_metrics:
        for weight in weight_decay:
            for rate in learning_rate:
                for bias_ in bias:
                    for non_lin_func in non_linearity:
                        extra_args = {"knn_metric": knn_metric, "loss_metric": nn.PairwiseDistance(p=2),
                                      "activation_function": non_lin_func}

                        args.l2 = weight
                        args.bias = bias_
                        args.learning_rate = rate

                        log.info("Starting training with: {}".format(extra_args))

                        log.debug("Building model...")
                        model = figet.Models.Model(args, vocabs, negative_samples, extra_args)

                        if len(args.gpus) >= 1:
                            model.cuda()

                        log.debug("Copying embeddings to model...")
                        model.init_params(word2vec, type2vec)
                        optim = figet.Optim(model.parameters(), args.learning_rate, args.max_grad_norm, args.l2)

                        nParams = sum([p.nelement() for p in model.parameters()])
                        log.debug("* number of parameters: %d" % nParams)

                        coach = figet.Coach(model, vocabs, train_data, dev_data, test_data, hard_test_data, optim, type2vec, args, extra_args)

                        # Train.
                        log.info("Start training...")
                        log.info(f"Activation: {non_lin_func}, knn_metric: {knn_metric}, Weight_decay: {weight}, learning_date: {rate}, bias: {bias_}")
                        ret = coach.train()
                        # log.info("Finish training with: {}".format(extra_args))
                        log.info(f"Activation: {non_lin_func}, knn_metric: {knn_metric}, Weight_decay: {weight}, learning_date: {rate}, bias: {bias_}")
                        log.info("Done!\n\n")


if __name__ == "__main__":
    main()
