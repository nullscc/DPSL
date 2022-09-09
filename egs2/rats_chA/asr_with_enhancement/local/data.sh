#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "$0 $*"
. utils/parse_options.sh

rm -rf data/test-clean
local/prepare_wcp_text_from_origin.sh /ssdhome/xzw521/data/librispeech/LibriSpeech/test-clean data/test-clean
rm -rf data/dev-clean
local/prepare_wcp_text_from_origin.sh /ssdhome/xzw521/data/librispeech/LibriSpeech/dev-clean data/dev-clean
rm -rf data/train-clean-100
local/prepare_wcp_text_from_origin.sh /ssdhome/xzw521/data/librispeech/LibriSpeech/train-clean-100 data/train-clean-100

python local/noisyspeech_synthesizer_multiprocessing.py --cfg noisyspeech_synthesizer_test_clean.cfg
python local/noisyspeech_synthesizer_multiprocessing.py --cfg noisyspeech_synthesizer_dev_clean.cfg
python local/noisyspeech_synthesizer_multiprocessing.py --cfg noisyspeech_synthesizer_train_clean_100.cfg

local/generate_train_file.sh data/test-clean data/test
local/generate_train_file.sh data/dev-clean data/dev
local/generate_train_file.sh data/train-clean-100 data/train
