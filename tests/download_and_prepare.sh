HOME_DIR=/rscratch/xihc/cache
HF_TOKEN=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

python download.py \
    --repo_id $1 \
    --hf_token $HF_TOKEN \
    --out_dir $HOME_DIR/checkpoints/$1

python convert_hf_checkpoint.py \
    --checkpoint_dir $HOME_DIR/checkpoints/$1