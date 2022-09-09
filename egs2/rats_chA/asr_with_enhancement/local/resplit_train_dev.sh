#!/usr/bin/env bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

dum_dir=$1
train_set=$2
dev_set=$3

all_dir=$dum_dir/train_dev
train_dir=$dum_dir/$train_set
dev_dir=$dum_dir/$dev_set
mkdir -p $all_dir 

filelist="spk1.scp text_spk1 utt2spk utt2num_samples wav.scp"
for f in $filelist; do
	cat $train_dir/$f > $all_dir/$f
	cat $dev_dir/$f >> $all_dir/$f
done
cat $train_dir/feats_type > $all_dir/feats_type
rm -rf $train_dir
mkdir -p $train_dir
rm -rf $dev_dir
mkdir -p $dev_dir

<$all_dir/wav.scp awk '{print $1}' > $all_dir/utt_list_file_bk
shuf $all_dir/utt_list_file_bk -o $all_dir/utt_list_file
all_num=`<${all_dir}/utt_list_file wc -l`
dev_num=$[$all_num/10]
train_num=$[$all_num-$dev_num]
head -n $train_num $all_dir/utt_list_file > $all_dir/train_utt_list_file
tail -n $dev_num $all_dir/utt_list_file > $all_dir/dev_utt_list_file

python local/subset_data_dir.py $all_dir/train_utt_list_file $all_dir $train_dir
python local/subset_data_dir.py $all_dir/dev_utt_list_file $all_dir $dev_dir
cat $all_dir/feats_type > $dev_dir/feats_type
cat $all_dir/feats_type > $train_dir/feats_type

utils/data/get_utt2dur.sh $dev_dir
utils/data/get_utt2dur.sh $train_dir
