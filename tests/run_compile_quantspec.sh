if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRITON_LOG_LEVEL=debug
PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
    --standalone \
    --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
    tests/quantspec_benchmark.py \
    --model /rscratch/xihc/cache/checkpoints/togethercomputer/llama-2-7b-32k/model.pth \
    --model_name togethercomputer/LLaMA-2-7B-32K \
    --marlin_path /rscratch/xihc/marlin/gptq/llama2_7b_32k_instruct_checkpoint.pt.marlin.g128 \
    --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
    --gamma 6 \
    --B 1 \
    --prefix_len 16384 \
    --gen_len 100 \
    --printout \
    --benchmark \
    --wq \
    --compile


    # --compile \
    # --wq
# TORCH_LOGS=+dynamo TORCHDYNAMO_VERBOSE=1 

