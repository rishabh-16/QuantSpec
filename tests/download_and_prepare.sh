HOME_DIR=/rscratch/rishabhtiwari/cache/
HF_TOKEN=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

python download.py \
    --repo_id $1 \
    --hf_token $HF_TOKEN \
    --out_dir $HOME_DIR/$1

python convert_hf_checkpoint.py \
    --checkpoint_dir $HOME_DIR/$1