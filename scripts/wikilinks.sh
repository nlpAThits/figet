#!/bin/bash

set -o errexit

# Data
corpus_name=wikilinks
corpus_dir=/hits/fast/nlp/lopezfo/projects/silverData/${corpus_name}
dataset_dir=${corpus_dir}

# Embeddings
embeddings_dir=data/embeddings
# embeddings=${embeddings_dir}/glove.840B.300d.txt
embeddings=${embeddings_dir}/miniglove.txt

# Checkpoints
ckpt=${corpus_dir}/ckpt
prep=${corpus_dir}/ckpt/prep
mkdir -p ${ckpt}

do_what=$1
prep_run=$2
run=$3

function get_current_run() {
    current_run=$2
    if [ -z "$current_run" ]; then    # empty
        all_runs=($(ls $1 | sort))
        last_run=0
        if [ -n "$all_runs" ]; then
            last_run=${all_runs[-1]}
        fi
        current_run="$(($last_run + 1))"
    fi
}

mkdir -p ${corpus_dir}

if [ "${do_what}" == "get_data" ];
then
    printf "\nDownloading corpus...`date`\n"
    if [ -d "${corpus_dir}/dataset" ]; then
        echo "Seems that you already have the dataset!"
    else
        wget http://www.cs.jhu.edu/~s.zhang/data/figet/${corpus_name}.zip -O ${corpus_dir}/dataset.zip
        (cd ${corpus_dir} && unzip dataset.zip && rm dataset.zip)
    fi

    printf "\nDownloading word embeddings...`date`\n"
    if [ -d "${embeddings_dir}" ]; then
        echo "Seems that you already have the embeddings!"
    else
        mkdir -p ${embeddings_dir}
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ${embeddings_dir}/embeddings.zip
        (cd ${embeddings_dir} && unzip embeddings.zip && rm embeddings.zip)
    fi

elif [ "${do_what}" == "preprocess" ];
then
    get_current_run $prep $prep_run
    prep=${prep}/${current_run}
    mkdir -p ${ckpt}
    mkdir -p ${prep}
    python2 -u ./preprocess.py \
        --train=${dataset_dir}/foo_train.jsonl --dev=${dataset_dir}/foo_dev.jsonl   \
        --test=${dataset_dir}/foo_test.jsonl \
        --use_doc=0 --word2vec=${embeddings} \
        --save_data=${prep}/${corpus_name} --shuffle

elif [ "${do_what}" == "train" ];
then
    get_current_run $prep $prep_run
    prep=${prep}/${current_run}
    get_current_run $ckpt $run
    ckpt=${ckpt}/${current_run}
    mkdir -p ${ckpt}
    python2 -u ./train.py \
        --data=${prep}/${corpus_name}.data.pt \
        --word2vec=${prep}/${corpus_name}.word2vec \
        --save_model=${ckpt}/${corpus_name}.model.pt \
        --save_tuning=${ckpt}/${corpus_name}.tuning.pt \
        --niter=-1 \
        --single_context=0 --use_hierarchy=0 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=2 --bias=0 --context_length=10

elif [ "${do_what}" == "adaptive-thres" ];
then
    python2 -u -m figet.adaptive_thres \
        --data=${ckpt}/${corpus_name}.tuning.pt \
        --optimal_thresholds=${ckpt}/${corpus_name}.thres

elif [ "${do_what}" == "inference" ];
then
    get_current_run $ckpt $run
    ckpt=${ckpt}/${current_run}
    python2 -u ./infer.py \
        --data=${dataset_dir}/foo_dev.jsonl \
        --save_model=${ckpt}/${corpus_name}.model.pt \
        --save_idx2threshold=${ckpt}/${corpus_name}.thres \
        --pred=${ckpt}/${corpus_name}.pred.txt \
        --single_context=0 --use_hierarchy=0 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=2 --bias=0 --context_length=10
fi
