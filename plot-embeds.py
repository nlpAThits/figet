
import torch
import argparse
import figet
import itertools
from tqdm import tqdm
from figet.Constants import TYPE_VOCAB
from coso.utils import export

parser = argparse.ArgumentParser("plot-embeds.py")

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

parser.add_argument('--file', help="model file with weights to process.")


args = parser.parse_args()

torch.cuda.set_device(0)
log = figet.utils.get_logging()

DATA = "/hits/basement/nlp/lopezfo/views/benultra/ckpt/prep/separate-knn/benultra"


def get_dataset(data, batch_size, key):
    dataset = data[key]
    dataset.set_batch_size(batch_size)
    return dataset


def main():
    log.debug("Loading data from '%s'." % DATA)
    data = torch.load(DATA + ".data.pt")
    vocabs = data["vocabs"]
    hierarchy = data["hierarchy"]

    dev_data = get_dataset(data, 512, "dev")
    test_data = get_dataset(data, 512, "test")

    # log.debug("Loading word2vecs")
    # word2vec = torch.load(DATA + ".word2vec")
    # log.debug("Loading type2vecs")
    # type2vec = torch.load(DATA + ".type2vec")

    state_dict = torch.load(args.file)

    proj_learning_rate = [0.05]         # not used
    proj_weight_decay = [0.0]           # not used
    proj_bias = [1]
    proj_hidden_layers = [1]
    proj_hidden_size = [500]
    proj_non_linearity = [None]         # not used
    proj_dropout = [0.3]                # not used

    k_neighbors = [4]
    args.knn_hyper = True

    cosine_factors = [50]               # not used
    hyperdist_factors = [1]             # not used

    args.type_dims = 10
    
    configs = itertools.product(proj_learning_rate, proj_weight_decay, proj_bias, proj_non_linearity, proj_dropout,
                                proj_hidden_layers, proj_hidden_size, cosine_factors, hyperdist_factors, k_neighbors)

    target_data = dev_data

    coarse_ids = vocabs[TYPE_VOCAB].get_coarse_ids()
    fine_ids = vocabs[TYPE_VOCAB].get_fine_ids()

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

        model.cuda()
        
        # log.debug("Copying embeddings to model...")
        # model.init_params(word2vec, type2vec)

        model.load_state_dict(state_dict)
        type2vec = model.type_lut.weight.data 

        log.info(f"Running model")
        model.eval()
        pred_and_true = []
        with torch.no_grad():
            for i in tqdm(range(len(target_data)), desc="run_model"):
                batch = target_data[i]
                types = batch[5]

                _, predicted_embeds, _, _, _, _, _ = model(batch, 0)

                for j in range(len(types)):
                    pred_and_true.append((predicted_embeds[0][j], predicted_embeds[1][j], predicted_embeds[2][j],
                                          types[j]))

        tensors, metadata = [], ["ids\tgran\tlabel\tis_pred\tcoarse_label"]
        labels = ["coarse", "fine", "ultrafine"]
        for i in range(len(pred_and_true)):
            item_id = f"id-{i}"

            # export types
            type_ids = pred_and_true[i][3].tolist()
            coarses = [i for i in type_ids if i in coarse_ids]
            fines = [i for i in type_ids if i in fine_ids and i not in coarse_ids]
            ultras = [i for i in type_ids if i not in fine_ids and i not in coarse_ids]
            coarse_label = vocabs[TYPE_VOCAB].idx2label[coarses[0]] if len(coarses) > 0 else "none"
            for gran, type_ids_by_gran in enumerate([coarses, fines, ultras]):
                for t_id in type_ids_by_gran:
                    tensors.append("\t".join(map(str, type2vec[t_id].tolist())))
                    true_label = vocabs[TYPE_VOCAB].idx2label[t_id]
                    metadata.append(f"{item_id}\t{labels[gran]}\t{true_label}\tfalse\tnone")

            # export predictions
            for j in range(3):
                tensors.append("\t".join(map(str, pred_and_true[i][j].tolist())))
                metadata.append(f"{item_id}\t{labels[j]}\tnone\ttrue\t{coarse_label}")


        path = f"img/plot/tensors-{args.file}"
        export(path + ".tsv", tensors)
        export(path + "-meta.tsv", metadata)


if __name__ == "__main__":
    main()
