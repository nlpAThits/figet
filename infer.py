#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import torch

import json
import preprocess
import figet
from figet.context_modules.doc2vec import Doc2Vec
from figet.utils import process_line, build_full_sentence
import figet.Constants as c


def interpret_attention(fields, attn, args):
    """
    :return: a sentence with the corresponding attention value next to each word
    """
    sent = []
    mention = fields[c.HEAD].split()
    left_context, right_context = (fields[c.LEFT_CTX].split(), fields[c.RIGHT_CTX].split())
    before_prev_context = left_context[:-args.context_length]
    after_right_context = right_context[args.context_length:]
    prev_context = left_context[-args.context_length:]
    next_context = right_context[:args.context_length]

    # before previous context
    for token in before_prev_context:
        sent.append("%s:%.2f" % (token, 0))

    for i, token in enumerate(prev_context):
        sent.append("%s:%.2f" % (token, attn[i]*100))

    sent += ["%s:%.2f" % (word, -1) for word in mention]

    for i, token in enumerate(next_context):
        sent.append("%s:%.2f" % (token, attn[-i-1]*100))

    # after next context
    for token in after_right_context:
        sent.append("%s:%.2f" % (token, 0))

    return " ".join(sent)


def dump_results(type_vocab, field_lines, preds, attns, args):
    ret = []
    if len(attns) == 0:
        attns = [None]*len(field_lines)       # :(

    for fields, (gold_type_, pred_type), attn in zip(field_lines, preds, attns):

        pred_type = list(sorted(map(type_vocab.get_label, pred_type)))
        sent = interpret_attention(fields, attn, args) if attn is not None else " ".join(build_full_sentence(fields))

        ret.append({
            "mention": fields[c.HEAD],
            "sent": sent,
            "prediction": pred_type,
            "gold": fields[c.TYPE],
        })

    with open(args.pred, "w", buffering=c.BUFFER_SIZE) as fp:
        for line in ret:
            fp.write(json.dumps(line) + "\n")


def read_data(data_file):
    return [process_line(line)[0] for line in open(data_file, buffering=c.BUFFER_SIZE)]


def main(args, log):
    # Load checkpoint.
    checkpoint = torch.load(args.save_model)
    vocabs, word2vec, states = checkpoint["vocabs"], checkpoint["word2vec"], checkpoint["states"]
    try:
        idx2threshold = torch.load(args.save_idx2threshold)
    except:
        idx2threshold = None
    log.info("Loaded checkpoint model from %s." %(args.save_model))

    # Load the pretrained model.
    model = figet.Models.Model(args, vocabs)
    model.load_state_dict(states)
    if len(args.gpus) >= 1:
        model.cuda()
    log.info("Init the model.")

    # Load data.
    data = preprocess.make_data(args.data, vocabs, args)

    i = 0
    total = len(data)
    for mention in data:
        mention.preprocess(vocabs, word2vec, args)
        i += 1
        if i % 100000 == 0:
            log.debug("Mentions processed: {} of {}".format(i, total))

    data = figet.Dataset(data, args, True)
    log.info("Loaded the data from %s." %(args.data))

    # Inference.
    preds, types, dists, attns = [], [], [], []
    model.eval()
    for i in range(len(data)):
        batch = data[i]
        loss, dist, attn = model(batch)
        preds += figet.adaptive_thres.predict(dist.data, batch[3].data, idx2threshold)
        types += [batch[3].data]
        dists += [dist.data]
        if attn is not None:
            attns += [attn.data]
    # types = torch.cat(types, 0).cpu().numpy()
    # dists = torch.cat(dists, 0).cpu().numpy()
    if len(attns) != 0:
        attns = torch.cat(attns, 0).cpu().numpy()
    log.info("Finished inference.")

    # Results.
    log.info("| Inference acc. %s |" % (figet.evaluate.evaluate(preds)))
    dump_results(vocabs["type"], read_data(args.data), preds, attns, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("infer.py")

    # Data options
    parser.add_argument("--data", required=True, type=str,
                        help="Data path.")
    parser.add_argument("--pred", required=True, type=str,
                        help="Prediction output.")
    parser.add_argument("--save_model", default="./save/model.pt", type=str,
                        help="Save the model.")
    parser.add_argument("--save_idx2threshold", default="./save/threshold.pt",
                        type=str, help="Save the model.")

    # Sentence-level context parameters
    parser.add_argument("--context_length", default=10, type=int,
                        help="Max length of the left/right context.")
    parser.add_argument("--context_input_size", default=300, type=int,
                        help="Input size of CotextEncoder.")
    parser.add_argument("--context_rnn_size", default=200, type=int,
                        help="RNN size of ContextEncoder.")
    parser.add_argument("--context_num_layers", default=1, type=int,
                        help="Number of layers of ContextEncoder.")
    parser.add_argument("--context_num_directions", default=2, choices=[1, 2], type=int,
                        help="Number of directions for ContextEncoder RNN.")
    parser.add_argument("--attn_size", default=100, type=int,
                        help=("Attention vector size."))
    parser.add_argument("--single_context", default=0, type=int,
                        help="Use single context.")

    # Other parameters
    parser.add_argument("--bias", default=0, type=int,
                        help="Whether to use bias in the linear transformation.")
    parser.add_argument("--dropout", default=0.5, type=float,
                        help="Dropout rate for all applicable modules.")
    parser.add_argument('--seed', type=int, default=3435,
                        help="Random seed")
    parser.add_argument('--shuffle', action="store_true",
                        help="Shuffle data.")

    # GPU
    parser.add_argument("--gpus", default=[], nargs="+", type=int,
                        help="Use CUDA on the listed devices.")

    args = parser.parse_args()

    if args.gpus:
        torch.cuda.set_device(args.gpus[0])

    figet.utils.set_seed(args.seed)
    log = figet.utils.get_logging()
    log.debug(args)

    main(args, log)
