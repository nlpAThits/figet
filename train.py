#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import torch
import random

import figet

parser = argparse.ArgumentParser("train.py")

# Data options
parser.add_argument("--data", required=True, type=str, help="Data path.")
parser.add_argument("--save_tuning", default="./save/tuning.pt", type=str,
                    help="Save the intermediate results for tuning.")
parser.add_argument("--save_model", default="./save/model.pt", type=str, help="Save the model.")

# Sentence-level context parameters
parser.add_argument("--context_length", default=10, type=int, help="Max length of the left/right context.")
parser.add_argument("--context_input_size", default=300, type=int, help="Input size of ContextEncoder.")
parser.add_argument("--context_rnn_size", default=200, type=int, help="RNN size of ContextEncoder.")
parser.add_argument("--context_num_layers", default=1, type=int, help="Number of layers of ContextEncoder.")
parser.add_argument("--context_num_directions", default=2, choices=[1, 2], type=int,
                    help="Number of directions for ContextEncoder RNN.")
parser.add_argument("--attn_size", default=100, type=int, help="Attention vector size.")
parser.add_argument("--single_context", default=0, type=int, help="Use single context.")

# Other parameters
parser.add_argument("--bias", default=0, type=int, help="Whether to use bias in the linear transformation.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Starting learning rate.")
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
parser.add_argument("--gpus", default=[], nargs="+", type=int, help="Use CUDA on the listed devices.")
parser.add_argument('--log_interval', type=int, default=100, help="Print stats at this interval.")

args = parser.parse_args()

if args.gpus:
    torch.cuda.set_device(args.gpus[0])

seed = random.randint(1, 10000)
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

    # Build model.
    log.debug("Building model...")
    model = figet.Models.Model(args, vocabs)
    optim = figet.Optim(args.learning_rate, args.max_grad_norm)

    if len(args.gpus) >= 1:
        model.cuda()
        figet.Dataset.GPUS = True

    log.debug("Loading word2vecs from '%s'." % args.word2vec)
    word2vec = torch.load(args.word2vec)

    model.init_params(word2vec)
    optim.set_parameters([p for p in model.parameters() if p.requires_grad])

    nParams = sum([p.nelement() for p in model.parameters()])
    log.debug("* number of parameters: %d" % nParams)

    coach = figet.Coach(model, vocabs, train_data, dev_data, test_data, optim, args)

    # Train.
    log.info("Start training...")
    ret = coach.train()

    # Save.
    tuning = {
        "type_vocab": vocabs["type"],
        "dev_dist": ret[1],
        "dev_type": ret[2],
        "test_dist": ret[3],
        "test_type": ret[4]
    }
    checkpoint = {
        "vocabs": vocabs,
        "word2vec": word2vec,
        "states": ret[0]
    }
    torch.save(tuning, args.save_tuning)
    torch.save(checkpoint, args.save_model)

    log.info("Done!")


if __name__ == "__main__":
    main()
