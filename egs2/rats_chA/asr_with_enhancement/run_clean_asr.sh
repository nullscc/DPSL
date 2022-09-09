#!/bin/bash
echo
echo "$0 $@"
echo
echo `date`
echo

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',`
set -e
set -u
set -o pipefail
stage=12  # stage1: prepared data, stage2: speed perturb, stage3: format wav.scp
stop_stage=12 # stage 10 is train stage stage11 inference stage and stage12 is score stage
ngpu=1
nj=16

# config
asr_config=conf/train_dpsl_asr.yaml
inference_config=conf/decode.yaml
lm_config=conf/train_lm_adam.yaml
audio_format=wav # Audio format (only in feats_type=raw).
# fs=8k
# train_set=rats_Ach_waveform_8k_train
# valid_set=rats_Ach_waveform_8k_valid
# test_sets=rats_Ach_waveform_8k_test
# dumpdir=dump_rats_Ach_waveform_8k     # Directory to dump features.
# expdir=exp_rats_Ach_waveform_8k_dpsl_asr       # Directory to save experiments.
fs=16k
train_set=train
valid_set=dev
test_sets=tclean
dumpdir=dump_libri100     # Directory to dump features.
expdir=exp_libri100      # Directory to save experiments.

use_lm=true
feats_type=raw
bpemode=bpe
feats_normalize=utterance_mvn

min_wav_duration=6
max_wav_duration=15

. utils/parse_options.sh

./clean_asr.sh   \
    --stage $stage \
    --stop_stage $stop_stage \
    --lang en \
    --feats_type $feats_type \
    --fs $fs \
    --audio_format $audio_format \
    --dumpdir "$dumpdir" \
    --expdir  "$expdir" \
    --ngpu $ngpu \
    --nbpe 1000 \
    --use_lm ${use_lm} \
    --bpemode $bpemode \
    --feats_normalize $feats_normalize \
	--min_wav_duration 6 \
    --max_wav_duration 15 \
    --joint_config  "${asr_config}" \
	--joint_exp exp_libri100/clean_asr_train_raw_bpe \
    --lm_config "${lm_config}" \
    --decode_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}"


