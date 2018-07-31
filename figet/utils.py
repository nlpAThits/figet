#!/usr/bin/env python
# encoding: utf-8


import sys
import logging
import random
import json
import numpy as np
import torch
import Constants


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def wc(files):
    if type(files) != list and type(files) != tuple:
        files = [files]
    return sum([sum([1 for _ in open(fp, buffering=Constants.BUFFER_SIZE)]) for fp in files])


def process_line(line):
    fields = json.loads(line)
    tokens = build_full_sentence(fields)
    return fields, tokens


def build_full_sentence(fields):
    return fields[Constants.LEFT_CTX].split() + fields[Constants.HEAD].split() + fields[Constants.RIGHT_CTX].split()


def to_sparse(tensor):
    """Given a one-hot encoding vector returns a list of the indexes with nonzero values"""
    return torch.nonzero(tensor)
