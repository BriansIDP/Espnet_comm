. ./path.sh || exit 1;

set="ihm_eval"
lm="no_lm"
# lm="external_wordlm_0.1"
# expdir=exp/ami_train_shift_transformer
# expdir=exp/ami_train_concatshift
expdir=exp/ami_train_transformer_concatshift_tune_12L
decode_dir=decode_${set}_${lm}
dict=data/lang_1char/ihm_train_units.txt
score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict} true
# score_crossutt.sh --wer true ${expdir}/${decode_dir} ${dict}
