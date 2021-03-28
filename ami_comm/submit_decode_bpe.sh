. ./path.sh || exit 1;

cmd="queue.pl -l qp=low,osrel=*,not_host='air120|air113|air112' -P black-svr"
# cmd="run.pl"
recog_set="ihm_dev ihm_eval"
# recog_set="ihm_eval"
nj=32
expdir=exp/ami_train_conformer_unigram200suffix_specaug
# expdir=exp/ami_train_conformer_KB_spec
tag="no_lm_b30"
pids=() # initialize pids
use_lm=false
use_wordlm=false
dumpdir=dump
do_delta=false
backend=pytorch

# Task specific
decode_config=conf/decode.yaml
recog_model=model.acc.best
dict=data/lang_1char/ihm_train_units.txt
lmexpdir=exp/train_rnnlm_pytorch_char_lm
wordlevel=false
fixedemb=false

bpemode=bpe
nbpe=200

for rtask in ${recog_set}; do
(
    decode_dir=decode_${rtask}_${tag}
    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
    splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    ${cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log decode_bpe.sh JOB ${rtask} \
        ${decode_dir} \
        ${feat_recog_dir} \
        ${expdir} \
        ${nj} \
        ${use_lm} \
        ${use_wordlm} \
        ${decode_config} \
        ${recog_model} \
        ${dict} \
        ${lmexpdir} \
        ${bpemode} \
        ${nbpe} \
        ${fixedemb} \
) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
echo "Finished"
