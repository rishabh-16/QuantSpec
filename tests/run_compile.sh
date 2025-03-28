# if [ $# -eq 0 ]; then
#     echo "Please provide the CUDA device number as an argument."
#     echo "Usage: $0 <cuda_device_number>"
#     exit 1
# fi
# export TORCH_USE_CUDA_DSA=1
# export CUDA_LAUNCH_BLOCKING=1
# export TRITON_LOG_LEVEL=debug

# PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
#         --standalone \
#         --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
#         tests/quantspec_benchmark.py \
#         --model /rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-Chat-128K/model.pth \
#         --model_name LargeWorldModel/LWM-Text-Chat-128K \
#         --marlin_path /rscratch/rishabhtiwari/QuantSpec_magidec/marlin/gptq/checkpoint.pt.marlin.g128 \
#         --dataset multilexsum \
#         --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
#         --gamma 6 \
#         --B 1 \
#         --prefix_len $2 \
#         --gen_len 90 \
#         --compile

model="/rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-Chat-128K/model.pth"
model_name="LargeWorldModel/LWM-Text-Chat-128K"
dataset="multilexsum"
marlin_path="/rscratch/rishabhtiwari/QuantSpec_magidec/marlin/gptq/lwm_text_chat_128k_checkpoint.pt.marlin.g128"

# model="/rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-128K/model.pth"
# model_name="LargeWorldModel/LWM-Text-128K"
# dataset="pg19"
# marlin_path="/rscratch/rishabhtiwari/QuantSpec_magidec/marlin/gptq/lwm_text_128k_checkpoint.pt.marlin.g128"

# model="/rscratch/rishabhtiwari/cache/togethercomputer/LLaMA-2-7B-32K/model.pth"
# model_name="togethercomputer/LLaMA-2-7B-32K"
# dataset="pg19"
# marlin_path="/rscratch/rishabhtiwari/marlin_updated/marlin/gptq/llama2_7b_32k_checkpoint.pt.marlin.g128"

# model="/rscratch/rishabhtiwari/cache/togethercomputer/Llama-2-7B-32K-Instruct/model.pth"
# model_name="togethercomputer/Llama-2-7B-32K-Instruct"
# dataset="multilexsum"
# marlin_path="/rscratch/rishabhtiwari/QuantSpec_magidec/marlin/gptq/llama2_7b_32k_instruct_checkpoint.pt.marlin.g128"



for prefix_len in 8192; do
  for gamma in 3 5; do
    for prefix_ratio in 0.25; do
      streamingllm_budget=$(printf "%.0f" $(echo "$prefix_len * $prefix_ratio" | bc))
      # if [ $prefix_len -gt 64000 ]; then
      #   gpus="4,5,6,7"
      if [ $prefix_len -gt 32000 ]; then
        gpus="6,7"
      else
        gpus="6"
      fi

      echo "----------------------------------------"
      echo "quantspec"
      echo "Prefix length: $prefix_len"
      echo "Gamma: $gamma"
      echo "----------------------------------------" 
      PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$gpus ENABLE_INTRA_NODE_COMM=1 torchrun \
        --standalone \
        --nproc_per_node=$(echo $gpus | tr ',' ' ' | wc -w) \
        tests/quantspec_benchmark.py \
        --model $model \
        --model_name $model_name \
        --marlin_path $marlin_path \
        --dataset $dataset \
        --rank_group $(seq -s ' ' 0 $(($(echo $gpus | tr ',' ' ' | wc -w) - 1))) \
        --gamma $gamma \
        --B 1 \
        --wq \
        --prefix_len $prefix_len \
        --gen_len 90 \
        --compile \
        --benchmark
    done
  done
done

# model="/rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-Chat-128K/model.pth"
# model_name="LargeWorldModel/LWM-Text-Chat-128K"
# dataset="multilexsum"


# for prefix_len in 128000 16000 32000 64000; do
#     if [ $prefix_len -gt 32000 ]; then
#         gpus="6,7"
#     else
#         gpus="6"
#     fi
#     # prefix_len=$((prefix_len-128))
#     echo "----------------------------------------" 
#     echo "autoregressive"
#     echo "Prefix length: $prefix_len"
#     echo "----------------------------------------" 
#     PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$gpus ENABLE_INTRA_NODE_COMM=1 torchrun \
#         --standalone \
#         --nproc_per_node=$(echo $gpus | tr ',' ' ' | wc -w) \
#         tests/baseline_benchmark.py \
#         --model /rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-Chat-128K/model.pth \
#         --model_name LargeWorldModel/LWM-Text-Chat-128K \
#         --dataset multilexsum \
#         --rank_group $(seq -s ' ' 0 $(($(echo $gpus | tr ',' ' ' | wc -w) - 1))) \
#         --B 1 \
#         --prefix_len $prefix_len \
#         --compile


    # echo "----------------------------------------" 
    # echo "quantspec w/o wq"
    # echo "Prefix length: $prefix_len"
    # echo "----------------------------------------" 
    # PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$gpus ENABLE_INTRA_NODE_COMM=1 torchrun \
    #     --standalone \
    #     --nproc_per_node=$(echo $gpus | tr ',' ' ' | wc -w) \
    #     tests/quantspec_benchmark.py \
    #     --model /rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-Chat-128K/model.pth \
    #     --model_name LargeWorldModel/LWM-Text-Chat-128K \
    #     --marlin_path /rscratch/rishabhtiwari/QuantSpec_magidec/marlin/gptq/checkpoint.pt.marlin.g128 \
    #     --dataset multilexsum \
    #     --rank_group $(seq -s ' ' 0 $(($(echo $gpus | tr ',' ' ' | wc -w) - 1))) \
    #     --gamma 6 \
    #     --B 1 \
    #     --prefix_len $prefix_len \
    #     --gen_len 90 \
    #     --compile
    
    # echo "----------------------------------------" 
    # echo "quantspec"
    # echo "Prefix length: $prefix_len"
    # echo "----------------------------------------" 
    # PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$gpus ENABLE_INTRA_NODE_COMM=1 torchrun \
    #     --standalone \
    #     --nproc_per_node=$(echo $gpus | tr ',' ' ' | wc -w) \
    #     tests/quantspec_benchmark.py \
    #     --model /rscratch/rishabhtiwari/cache/LargeWorldModel/LWM-Text-Chat-128K/model.pth \
    #     --model_name LargeWorldModel/LWM-Text-Chat-128K \
    #     --marlin_path /rscratch/rishabhtiwari/QuantSpec_magidec/marlin/gptq/checkpoint.pt.marlin.g128 \
    #     --dataset multilexsum \
    #     --rank_group $(seq -s ' ' 0 $(($(echo $gpus | tr ',' ' ' | wc -w) - 1))) \
    #     --gamma 6 \
    #     --B 1 \
    #     --wq \
    #     --prefix_len $prefix_len \
    #     --gen_len 90 \
    #     --compile
# done
    
# #    --compile \
# # TORCH_LOGS=+dynamo TORCHDYNAMO_VERBOSE=1 
# #     --marlin_path /rscratch/rishabhtiwari/marlin_updated/marlin/gptq/llama2_7b_32k_checkpoint.pt.marlin.g128 \
