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
src_dir=$1
dest_dir=$2

root_dir="$(realpath .)"
mkdir -p $dest_dir
file_list="spk2gender spk2utt text utt2spk wav.scp"
for f in $file_list; do
	cp $src_dir/$f $dest_dir
done
mv $dest_dir/wav.scp $dest_dir/wav.scp.tmp
mv $dest_dir/text $dest_dir/text_spk1
find $root_dir/$src_dir/noisy -name "*.wav" | sort -u > $dest_dir/flist
paste -d' ' <(<$dest_dir/wav.scp.tmp awk '{print $1}') <(<$dest_dir/flist awk '{print $1}') > $dest_dir/wav.scp
rm -f $dest_dir/flist $dest_dir/wav.scp.tmp
find $root_dir/$src_dir/clean -name "*.wav" | sort -u > $dest_dir/flist
paste -d' ' <(<$dest_dir/wav.scp awk '{print $1}') <(<$dest_dir/flist awk '{print $1}') > $dest_dir/spk1.scp
rm -f $dest_dir/flist $dest_dir/wav.scp.tmp

