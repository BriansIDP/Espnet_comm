. ./path.sh || exit 1;
export PYTHONPATH="/home/dawna/gs534/espnet/egs/ami/asr1/exp/external_rnnlm:$PYTHONPATH"
export PYTHONPATH="/home/dawna/gs534/espnet/egs/ami/asr1/exp/external_rnnlm_crossutt:$PYTHONPATH"
export PYTHONPATH="/home/dawna/gs534/espnet/espnet/nets/pytorch_backend/lm:$PYTHONPATH"
echo "Start Decoding"
echo "pythonpath = $PYTHONPATH"
nj=1
recog_set="ihm_dev"
use_lm=false
use_wordlm=false
dumpdir=dump
do_delta=false
backend=pytorch

# Task specific
# lmexpdir=exp/external_memorynet
# lmexpdir=exp/external_rnnlm
# lmexpdir=exp/train_rnnlm_pytorch_char_lm
# lmexpdir=exp/train_rnnlm_pytorch_default_wordlm
# lmexpdir=exp/external_rnnlm_crossutt
tag="debug"
# expdir=exp/ami_train_transformer_concat
# expdir=exp/ami_train_conformer_unigram200suffix_specaug
expdir=exp/ami_train_conformer_KB_spec
recog_model=model.acc.best
# decode_config=conf/decode_v2.yaml
decode_config=conf/decode.yaml
# decode_config=conf/decode_crossutt.yaml
# dict=data/lang_1char/ihm_train_units.txt
bpemode=unigram
nbpe=200
suffix='suffix'
dict=data/lang_char/ihm_train_${bpemode}${nbpe}${suffix}_units.txt

pids=() # initialize pids
for rtask in ${recog_set}; do
(
    decode_dir=decode_${rtask}_${tag}

    if [ ${use_lm} = true ]; then
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
    else
        echo "No language model is involved."
        recog_opts=""
    fi

    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

    # split data
    # splitjson.py --parts ${nj} ${feat_recog_dir}/data_context.json

    #### use CPU for decoding
    ngpu=0

    asr_recog.py \
        --config ${decode_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --recog-json ${feat_recog_dir}/split1utt/data_suffix.1.json \
        --result-label ${expdir}/${decode_dir}/data.1.json \
        --model ${expdir}/results/${recog_model}  \
        ${recog_opts} \
        # --external \
        # --suboperation concatshift \
        # --crossutt memorynet \
        # --cont \
        # --crossutt \
)
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
