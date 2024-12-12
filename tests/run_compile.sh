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
    --model /home/rishabhtiwari/cache/checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --marlin_path /home/rishabhtiwari/cache/marlin/llama3_8b_checkpoint.pt.marlin.g128 \
    --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
    --gamma 6 \
    --B 1 \
    --prefix_len 16000 \
    --gen_len 64 \
    --printout \
    --compile \
    --benchmark

#    --compile \
# TORCH_LOGS=+dynamo TORCHDYNAMO_VERBOSE=1 