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
    tests/baseline_benchmark.py \
    --model /rscratch/xihc/cache/checkpoints/togethercomputer/llama-2-7b-32k/model.pth \
    --model_name togethercomputer/LLaMA-2-7B-32K \
    --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
    --B 1 \
    --prefix_len 16384 \
    --gen_len 100 \
    --printoutput \
    --compile
