import sys
utt_list_file = sys.argv[1]
src_dir = sys.argv[2]
dest_dir = sys.argv[3]

utt_set = set()
with open(utt_list_file, "r") as f:
    for line in f.readlines():
        item = line.strip()
        utt_set.add(item)

handle_files = "spk1.scp text_spk1 utt2spk utt2num_samples wav.scp"
for filename in handle_files.split(" "):
    with open(f"{src_dir}/{filename}", "r") as r_f:
        with open(f"{dest_dir}/{filename}", "w+") as w_f:
            for line in r_f.readlines():
                utt = line.split(" ")[0]
                if utt not in utt_set: continue
                w_f.write(line.strip() + "\n")
