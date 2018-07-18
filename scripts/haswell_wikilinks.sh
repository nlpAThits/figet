#!/bin/bash

set -o errexit

# Data
corpus_name=wikilinks
corpus_dir=/hits/basement/nlp/lopezfo/data/${corpus_name}
dataset_dir=${corpus_dir}

tenk_corpus_name=tenk_wikilinks
tenk_corpus_dir=/hits/basement/nlp/lopezfo/views/${corpus_name}/${tenk_corpus_name}
tenk_dataset_dir=${tenk_corpus_dir}

onem_corpus_name=onem_wikilinks
onem_corpus_dir=/hits/basement/nlp/lopezfo/views/${corpus_name}/${onem_corpus_name}
onem_dataset_dir=${onem_corpus_dir}

# Embeddings
embeddings_dir=data/embeddings
embeddings=${embeddings_dir}/glove.840B.300d.txt

# Checkpoints
ckpt=${corpus_dir}/ckpt
tenk_ckpt=${tenk_corpus_dir}/ckpt
onem_ckpt=${onem_corpus_dir}/ckpt

do_what=$1
run=$2  # mandatory in case of training

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

elif [ "${do_what}" == "preprocess_tenk" ];
then
    get_current_run $tenk_ckpt $run
    tenk_ckpt=${tenk_ckpt}/${current_run}
    mkdir -p ${tenk_ckpt}
    python2 -u ./preprocess.py \
        --train=${tenk_dataset_dir}/train.jsonl --dev=${tenk_dataset_dir}/dev.jsonl   \
        --test=${tenk_dataset_dir}/test.jsonl \
        --use_doc=0 --word2vec=${embeddings} \
        --save_data=${tenk_ckpt}/${tenk_corpus_name} --shuffle

elif [ "${do_what}" == "train_tenk" ];
then
    get_current_run $tenk_ckpt $run
    tenk_ckpt=${tenk_ckpt}/${current_run}
    python2 -u ./train.py \
        --data=${tenk_ckpt}/${tenk_corpus_name}.data.pt \
        --word2vec=${tenk_ckpt}/${tenk_corpus_name}.word2vec \
        --save_model=${tenk_ckpt}/${tenk_corpus_name}.model.pt \
        --save_tuning=${tenk_ckpt}/${tenk_corpus_name}.tuning.pt \
        --niter=-1 \
        --gpus=0 \
        --single_context=0 --use_hierarchy=0 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=2 --bias=0 --context_length=10

elif [ "${do_what}" == "preprocess_onem" ];
then
    get_current_run $onem_ckpt $run
    onem_ckpt=${onem_ckpt}/${current_run}
    mkdir -p ${onem_ckpt}
    python2 -u ./preprocess.py \
        --train=${onem_dataset_dir}/train.jsonl --dev=${onem_dataset_dir}/dev.jsonl   \
        --test=${onem_dataset_dir}/test.jsonl \
        --use_doc=0 --word2vec=${embeddings} \
        --save_data=${onem_ckpt}/${onem_corpus_name} --shuffle

elif [ "${do_what}" == "train_onem" ];
then
    get_current_run $onem_ckpt $run
    onem_ckpt=${onem_ckpt}/${current_run}
    python2 -u ./train.py \
        --data=${onem_ckpt}/${onem_corpus_name}.data.pt \
        --word2vec=${onem_ckpt}/${onem_corpus_name}.word2vec \
        --save_model=${onem_ckpt}/${onem_corpus_name}.model.pt \
        --save_tuning=${onem_ckpt}/${onem_corpus_name}.tuning.pt \
        --niter=-1 \
        --gpus=0 \
        --single_context=0 --use_hierarchy=0 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=2 --bias=0 --context_length=10


elif [ "${do_what}" == "preprocess" ];
then
    get_current_run $ckpt $run
    ckpt=${ckpt}/${current_run}
    mkdir -p ${ckpt}
    python2 -u ./preprocess.py \
        --train=${dataset_dir}/train.jsonl --dev=${dataset_dir}/sub_dev.jsonl   \
        --test=${dataset_dir}/sub_test.jsonl \
        --use_doc=0 --word2vec=${embeddings} \
        --save_data=${ckpt}/${corpus_name} --shuffle

elif [ "${do_what}" == "train" ];
then
    get_current_run $ckpt $run
    ckpt=${ckpt}/${current_run}
    python2 -u ./train.py \
        --data=${ckpt}/${corpus_name}.data.pt \
        --word2vec=${ckpt}/${corpus_name}.word2vec \
        --save_model=${ckpt}/${corpus_name}.model.pt \
        --save_tuning=${ckpt}/${corpus_name}.tuning.pt \
        --niter=-1 \
        --gpus=1 \
        --single_context=0 --use_hierarchy=0 --epochs=45 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=2 --bias=0 --context_length=10

elif [ "${do_what}" == "adaptive-thres" ];
then
    get_current_run $ckpt $run
    ckpt=${ckpt}/${current_run}
    python2 -u -m figet.adaptive_thres \
        --data=${ckpt}/${corpus_name}.tuning.pt \
        --optimal_thresholds=${ckpt}/${corpus_name}.thres

elif [ "${do_what}" == "inference" ];
then
    get_current_run $ckpt $run
    ckpt=${ckpt}/${current_run}
    python2 -u ./infer.py \
        --data=${dataset_dir}/test.jsonl \
        --save_model=${ckpt}/${corpus_name}.model.pt \
        --save_idx2threshold=${ckpt}/${corpus_name}.thres \
        --pred=${ckpt}/${corpus_name}.pred.txt \
        --gpus=0 \
        --single_context=0 --use_hierarchy=0 \
        --use_doc=0 --use_manual_feature=0 \
        --context_num_layers=2 --bias=0 --context_length=10
fi

