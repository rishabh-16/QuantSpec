if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

TORCH_LOGS=+dynamo TORCHDYNAMO_VERBOSE=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
    --standalone \
    --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
    tests/quantspec_benchmark.py \
    --model /rscratch/xihc/MagicDec/checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth \
    --model_name meta-llama/Meta-Llama-3.1-8B \
    --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
    --gamma 3 \
    --B 1 \
    --prefix_len 16000 \
    --gen_len 64 \
    --printout \
    --compile \
    --benchmark

#    --compile \
