#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
from tqdm import tqdm
import torch

import figet
from figet.Constants import TOKEN_VOCAB, TYPE_VOCAB, BUFFER_SIZE, TYPE
from figet.utils import process_line

log = figet.utils.get_logging()


def make_vocabs(args):
    """
    It creates a Dict for the words on the whole dataset, and the types
    """
    token_vocab = figet.Dict([figet.Constants.PAD_WORD, figet.Constants.UNK_WORD], lower=args.lower)
    type_vocab = figet.Dict()

    all_files = (args.train, args.dev, args.test)
    bar = tqdm(desc="make_vocabs", total=figet.utils.wc(all_files))
    for data_file in all_files:
        for line in open(data_file, buffering=BUFFER_SIZE):
            bar.update()

            fields, tokens = process_line(line)

            for token in tokens:
                token_vocab.add(token)

            for type_ in fields[TYPE]:
                type_vocab.add(type_)

    bar.close()

    log.info("Created vocabs:\n\t#token: %d\n\t#type: %d" % (token_vocab.size(), type_vocab.size()))

    return {TOKEN_VOCAB: token_vocab, TYPE_VOCAB: type_vocab}


def make_word2vec(filepath, vocab):
    word2vec = figet.Word2Vec()
    log.info("Start loading pretrained word vecs")
    for line in tqdm(open(filepath), total=figet.utils.wc(filepath)):
        fields = line.strip().split()
        token = fields[0]
        vec = list(map(float, fields[1:]))
        word2vec.add(token, torch.Tensor(vec))

    ret = []
    oov = 0
    unk_vec = word2vec.get_unk_vector()

    for idx in xrange(vocab.size()):
        token = vocab.idx2label[idx]
        if token == figet.Constants.PAD_WORD:
            ret.append(torch.zeros(unk_vec.size()))
            continue

        if token in word2vec:
            vec = word2vec.get_vec(token)
        else:
            oov += 1
            vec = unk_vec
        ret.append(vec)             # Here it appends n (with n ~ 0.66 * token.size()) times the unk vec
    ret = torch.stack(ret)          # creates a "matrix" of token.size() x embed_dim
    log.info("* OOV count: %d" %oov)
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

    if args.shuffle:    # True by default               # SI los voy a ordenar, que sea por el largo del contexto, no el largo general
        log.info('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        data = [data[idx] for idx in perm]

    log.info("Prepared %d mentions.".format(count))

    dataset = figet.Dataset(data, args.batch_size, args)
    dataset.to_matrix(vocabs, word2vec, args)

    return dataset


def main(args):

    log.info("Preparing vocabulary...")
    vocabs = make_vocabs(args)

    log.info("Preparing pretrained word vectors...")
    word2vec = make_word2vec(args.word2vec, vocabs["token"])

    log.info("Preparing training...")
    train = make_data(args.train, vocabs, word2vec, args)
    log.info("Preparing dev...")
    dev = make_data(args.dev, vocabs, word2vec, args)
    log.info("Preparing test...")
    test = make_data(args.test, vocabs, word2vec, args)

    log.info("Saving pretrained word vectors to '%s'..." % (args.save_data + ".word2vec"))
    torch.save(word2vec, args.save_data + ".word2vec")

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
    parser.add_argument("--emb_size", default=300, type=int, help="Embedding size.")

    # Context
    parser.add_argument("--context_length", default=10, type=int, help="Max length of the left/right context.")
    parser.add_argument("--single_context", default=0, type=int, help="Use single context.")

    # Ops
    parser.add_argument("--use_doc", default=0, type=int, help="Whether to use the doc context or not.")
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data.")
    parser.add_argument('--seed', type=int, default=3435, help="Random seed")
    parser.add_argument('--lower', action='store_true', help='lowercase data')

    # Output data
    parser.add_argument("--save_data", required=True, help="Path to the output data.")
    parser.add_argument("--save_doc2vec", help="Path to the doc2vec model.")

    args = parser.parse_args()

    figet.utils.set_seed(args.seed)

    main(args)
