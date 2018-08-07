#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
from tqdm import tqdm
import torch

import figet
from figet.Constants import TOKEN_VOCAB, TYPE_VOCAB, BUFFER_SIZE, TYPE
from figet.utils import process_line, clean_type

log = figet.utils.get_logging()


def make_vocabs(args):
    """
    It creates a Dict for the words on the whole dataset, and the types
    """
    token_vocab = figet.TokenDict(lower=args.lower)
    type_vocab = figet.Dict()

    all_files = (args.train, args.dev, args.test)
    bar = tqdm(desc="make_vocabs", total=figet.utils.wc(all_files))
    for data_file in all_files:
        for line in open(data_file, buffering=BUFFER_SIZE):
            bar.update()

            fields, tokens = process_line(line)

            for token in tokens:
                token_vocab.add(token)

            ############ For now, I only use the first type, in case that there is more than one ##########
            mention_type = clean_type(fields[TYPE][0])
            type_vocab.add(mention_type)

    bar.close()

    log.info("Created vocabs:\n\t#token: %d\n\t#type: %d" % (token_vocab.size(), type_vocab.size()))

    return {TOKEN_VOCAB: token_vocab, TYPE_VOCAB: type_vocab}


def make_word2vec(filepath, tokenDict):
    word2vec = figet.Word2Vec()
    log.info("Start loading pretrained word vecs")
    for line in tqdm(open(filepath), total=figet.utils.wc(filepath)):
        fields = line.strip().split()
        token = fields[0]
        try:
            vec = list(map(float, fields[1:]))
        except ValueError:
            continue
        word2vec.add(token, torch.Tensor(vec))

    ret = []
    oov = 0

    # PAD word (index 0) is a vector full of zeros
    ret.append(torch.zeros(word2vec.get_unk_vector().size()))
    tokenDict.label2wordvec_idx[figet.Constants.PAD_WORD] = 0

    for idx in range(1, tokenDict.size()):
        token = tokenDict.idx2label[idx]

        if token in word2vec:
            vec = word2vec.get_vec(token)
            tokenDict.label2wordvec_idx[token] = len(ret)
            ret.append(vec)
        else:
            oov += 1

    ret = torch.stack(ret)          # creates a "matrix" of token.size() x embed_dim
    log.info("* OOV count: %d" %oov)
    log.info("* Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    return ret


def make_type2vec(filepath, typeDict):
    type2vec = figet.Word2Vec()
    log.info("Start loading pretrained type vecs")
    for line in tqdm(open(filepath), total=figet.utils.wc(filepath)):
        fields = line.strip().split()
        mention_type = fields[0]
        try:
            vec = list(map(float, fields[1:]))
        except ValueError:
            continue
        type2vec.add(mention_type, torch.Tensor(vec))

    ret = []

    for idx in range(typeDict.size()):
        mention_type = typeDict.idx2label[idx]
        vec = type2vec.get_vec(mention_type)
        ret.append(vec)

    ret = torch.stack(ret)          # creates a "matrix" of typeDict.size() x type_embed_dim
    log.info("* Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    return ret


def make_data(data_file, vocabs, word2vec, args):
    """
    :param data_file: train, dev or test
    :param vocabs:
    :param args:
    :return:
    """
    count = 0
    data, sizes = [], []
    for line in tqdm(open(data_file, buffering=BUFFER_SIZE), total=figet.utils.wc(data_file)):
        fields, tokens = process_line(line)

        mention = figet.Mention(fields)
        data.append(mention)
        sizes.append(len(tokens))
        count += 1

    if args.shuffle:    # True by default
        log.info('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        data = [data[idx] for idx in perm]

    log.info("Prepared %d mentions.".format(count))
    dataset = figet.Dataset(data, args)

    log.info("Transforming to matrix {} mentions from {} ".format(count, data_file))
    dataset.to_matrix(vocabs, word2vec, args)

    return dataset


def main(args):

    log.info("Preparing vocabulary...")
    vocabs = make_vocabs(args)

    log.info("Preparing pretrained word vectors...")
    word2vec = make_word2vec(args.word2vec, vocabs[TOKEN_VOCAB])

    log.info("Preparing pretrained type vectors...")
    type2vec = make_type2vec(args.type2vec, vocabs[TYPE_VOCAB])

    log.info("Preparing training...")
    train = make_data(args.train, vocabs, word2vec, args)
    log.info("Preparing dev...")
    dev = make_data(args.dev, vocabs, word2vec, args)
    log.info("Preparing test...")
    test = make_data(args.test, vocabs, word2vec, args)

    log.info("Saving pretrained word vectors to '%s'..." % (args.save_data + ".word2vec"))
    torch.save(word2vec, args.save_data + ".word2vec")

    log.info("Saving pretrained type vectors to '%s'..." % (args.save_data + ".type2vec"))
    torch.save(type2vec, args.save_data + ".type2vec")

    log.info("Saving data to '%s'..." % (args.save_data + ".data.pt"))
    save_data = {"vocabs": vocabs, "train": train, "dev": dev, "test": test}
    torch.save(save_data, args.save_data + ".data.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")

    # Input data
    parser.add_argument("--train", required=True, help="Path to the training data.")
    parser.add_argument("--dev", required=True, help="Path to the dev data.")
    parser.add_argument("--test", required=True, help="Path to the test data.")
    parser.add_argument("--word2vec", default="", type=str, help="Path to pretrained word vectors.")
    parser.add_argument("--type2vec", default="", type=str, help="Path to pretrained type vectors.")
    parser.add_argument("--emb_size", default=300, type=int, help="Embedding size.")

    # Context
    parser.add_argument("--context_length", default=10, type=int, help="Max length of the left/right context.")
    parser.add_argument("--single_context", default=0, type=int, help="Use single context.")

    # Ops
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data.")
    parser.add_argument('--seed', type=int, default=3435, help="Random seed")
    parser.add_argument('--lower', action='store_true', help='lowercase data')

    # Output data
    parser.add_argument("--save_data", required=True, help="Path to the output data.")
    parser.add_argument("--save_doc2vec", help="Path to the doc2vec model.")

    args = parser.parse_args()

    figet.utils.set_seed(args.seed)

    main(args)
