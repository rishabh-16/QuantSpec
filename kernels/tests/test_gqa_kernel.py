import math
import torch
import unittest
import numpy as np

from flashdecoding.fp16_gqa.gqa_flash_decoding import token_decode_gqa_attention_flash_decoding

from quantize.quant_pack_int4toint32 import triton_int4toint32_quantize_and_pack_along_last_dim, triton_int4toint32_quantize_and_pack_along_penultimate_dim
from quantize.quant_pack_int4toint8 import triton_int4toint8_quantize_along_penultimate_dim_and_pack_along_last_dim
from quantize.quant_pack_int8toint8 import triton_int8toint8_quantize_along_penultimate_dim_and_pack_along_last_dim
from quantize.quant_pack_int8toint8_upperlower import triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim

from tests.test_utils import quantize_tensor_last_dim, quantize_tensor_penultimate_dim

group_size = 128
bit = 4
batch_size, num_heads, kv_heads, sequence_length, head_dim = 1, 32, 8, 128 * 1024, 128
cache_len, residual_len = 128 * 1024, 272

num_runs = 20

def _init_simulate_kvcache(group_size, bit, B, nh, kv_nh, T, D):
    init_method = "random"
    dtype, device = torch.float16, "cuda"

    if init_method == "random":
        query_states = torch.rand((B, nh, 1, D), device=device, dtype=dtype)
        key_states = torch.rand((B, kv_nh, T, D), device=device, dtype=dtype)
        value_states = torch.rand((B, kv_nh, T, D), device=device, dtype=dtype)

    if init_method == "saved":
        query_states, key_states, value_states, dropout, softmax_scale, causal = \
            torch.load(f"debug_tensor/debug_llama_bs1_bit4_sl131072.pt", weights_only=True)

        # Hack the input # [B, T, nh, D]
        query_states, key_states, value_states = query_states[:B, :T, :nh, :head_dim], key_states[:B, :T, :kv_nh, :head_dim].contiguous(), value_states[:B, :T, :kv_nh, :head_dim].contiguous()
        
        # Transpose and permute to align the input of flashattention with our implementation
        query_states, key_states, value_states = query_states.transpose(1, 2).contiguous(), key_states.transpose(1, 2).contiguous(), value_states.transpose(1, 2).contiguous()

    return query_states, key_states, value_states

"""
Simulated MHA
"""
def _test_simulation_mha(query_states, key_states, value_states, B, nh, T, D):
    from flash_attn import flash_attn_func
    
    q = query_states.transpose(1, 2)  # (B, nh, T, D)
    k = key_states.transpose(1, 2)    # (B, nh_kv, T, D)
    v = value_states.transpose(1, 2)  # (B, nh_kv, T, D)
    
    output_no_quantize = flash_attn_func(q, k, v, causal=False)

    return output_no_quantize

def _test_simulation_verify_query_mha(longer_query_states, key_states, value_states, B, nh, T, D):
    attn_weights = torch.matmul(longer_query_states, key_states.transpose(2, 3)) / math.sqrt(D)

    query_sequence_length, full_k_seqlen = longer_query_states.shape[2], key_states.shape[2]
    q_head_num = longer_query_states.shape[1]
    attn_mask = torch.tril(torch.ones((query_sequence_length, full_k_seqlen), device=longer_query_states.device), diagonal=full_k_seqlen-query_sequence_length)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, q_head_num, 1, 1)
    attn_weights[attn_mask == 0] = float('-inf')

    attn_output = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(longer_query_states.dtype)
    output_no_quantize = torch.matmul(attn_output, value_states)

    return output_no_quantize

def _test_fp16_gqa_flash_decoding(query_states, key_states, value_states, cache_len):
    output = token_decode_gqa_attention_flash_decoding(query_states, key_states, value_states, qcache_len=cache_len)
    return output


def measure_cuda_timing(test_func, num_runs=20, *args, **kwargs):
    """
    Measures the execution time of a CUDA function in milliseconds.
    
    Parameters:
        test_func (callable): The function to test, which performs CUDA operations.
        num_runs (int): Number of times to run the function for timing. Default is 10.
        *args: Positional arguments to pass to the test function.
        **kwargs: Keyword arguments to pass to the test function.

    Returns:
        float: Median execution time in milliseconds.
    """
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        start_events[i].record()
        test_func(*args, **kwargs)
        end_events[i].record()
    
    torch.cuda.synchronize()  # Wait for all events to complete
    elapsed_times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]
    
    return np.median(elapsed_times)


class TestMHA(unittest.TestCase):
    def test_mha(self):
        query_states, key_states, value_states = _init_simulate_kvcache(group_size, bit, batch_size, num_heads, kv_heads, sequence_length, head_dim)
        simulated_output = _test_simulation_mha(query_states, key_states, value_states, \
                                                batch_size, num_heads, sequence_length, head_dim)
        
        _value_states = value_states.transpose(2, 3).contiguous()
        fp16_output = _test_fp16_gqa_flash_decoding(query_states, key_states, _value_states, cache_len=cache_len)
        median_time = measure_cuda_timing(
            _test_fp16_gqa_flash_decoding,
            20, 
            query_states, key_states, _value_states, cache_len=cache_len
        )
        print(f"Median execution time of Flash Decoding FP16: {median_time} ms")

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()