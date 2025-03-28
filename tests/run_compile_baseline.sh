# if [ $# -eq 0 ]; then
#     echo "Please provide the CUDA device number as an argument."
#     echo "Usage: $0 <cuda_device_number>"
#     exit 1
# fi
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRITON_LOG_LEVEL=debug

  
  # PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
  #   --standalone \
  #   --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
  #   tests/baseline_benchmark.py \
  #   --model /rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-Chat-128K/model.pth \
  #   --model_name LargeWorldModel/LWM-Text-Chat-128K \
  #   --dataset multilexsum \
  #   --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
  #   --B 1 \
  #   --prefix_len $2 \
  #   --gen_len 90 \
  #   --compile 

  PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
    --standalone \
    --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
    tests/baseline_benchmark.py \
    --model /rscratch/rishabhtiwari/cache/meta-llama/Llama-3.1-8B/model.pth \
    --model_name meta-llama/Llama-3.1-8B \
    --dataset multilexsum \
    --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
    --B 1 \
    --prefix_len $2 \
    --gen_len 90 \
    --compile
    # PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
#     --standalone \
#     --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
#     tests/baseline_benchmark.py \
#     --model /home/rishabhtiwari/cache/checkpoints/meta-llama/Llama-2-7b-hf/model.pth \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
#     --B 1 \
#     --prefix_len 10000 \
#     --gen_len 64 \
#     --compile \
#     --printout
