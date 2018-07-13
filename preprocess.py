#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
from tqdm import tqdm
import torch

import figet
from figet.context_modules.doc2vec import Doc2Vec
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


def make_data(data_file, vocabs, args, doc2vec=None):
    """
    :param data_file: train, dev or test
    :param vocabs:
    :param args:
    :param doc2vec: None by default (by default I mean on the scripts)
    :return:
    """
    count, ignored = 0, 0
    data, sizes = [], []
    for line in tqdm(open(data_file, buffering=BUFFER_SIZE), total=figet.utils.wc(data_file)):
        fields, tokens = process_line(line)

        doc_vec = None
        # if args.use_doc == 1:       # 0 by default
        #     if len(fields) == 5:
        #         doc = fields[2]
        #     else:
        #         doc = fields[7].replace('\\n', ' ').strip()
        #     doc_vec = doc2vec.transform(doc)

        mention = figet.Mention(fields, doc_vec)
        data.append(mention)
        sizes.append(len(tokens))
        count += 1

    if args.shuffle:    # True by default
        log.info('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        data = [data[idx] for idx in perm]

    log.info("Prepared %d mentions (%d ignored due to malformed input.)" %(count, ignored))

    return data


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
        ret.append(vec)
    ret = torch.stack(ret)          # creates a "matrix" of token.size() x embed_dim
    log.info("* OOV count: %d" %oov)
    log.info("* Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    return ret


def main(args):

    doc2vec = None
    if args.use_doc == 1:       # it is 0 by default
        doc2vec = Doc2Vec(save_path=args.save_doc2vec)
        doc2vec.load()

    log.info("Preparing vocabulary...")
    vocabs = make_vocabs(args)

    log.info("Preparing training...")
    train = make_data(args.train, vocabs, args, doc2vec)
    log.info("Preparing dev...")
    dev = make_data(args.dev, vocabs, args, doc2vec)
    log.info("Preparing test...")
    test = make_data(args.test, vocabs, args, doc2vec)

    log.info("Preparing pretrained word vectors...")
    word2vec = make_word2vec(args.word2vec, vocabs["token"])
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

    # Ops
    parser.add_argument("--use_doc", default=0, type=int, help="Whether to use the doc context or not.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data.")
    parser.add_argument('--seed', type=int, default=3435, help="Random seed")
    parser.add_argument('--lower', action='store_true', help='lowercase data')

    # Output data
    parser.add_argument("--save_data", required=True, help="Path to the output data.")
    parser.add_argument("--save_doc2vec", help="Path to the doc2vec model.")

    args = parser.parse_args()

    figet.utils.set_seed(args.seed)

    main(args)
