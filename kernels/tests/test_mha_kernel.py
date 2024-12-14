import math
import torch
import unittest
import numpy as np

from flashdecoding.fp16.flash_decoding import token_decode_attention_flash_decoding
from flashdecoding.int4_int32pack.int4kv_int32pack_flash_decoding import token_decode_attention_int4kv_int32pack_flash_decoding
from flashdecoding.int4_int8pack.int4kv_int8pack_flash_decoding import token_decode_attention_int4kv_int8pack_flash_decoding
from flashdecoding.int8_int8pack.int8kv_int8pack_flash_decoding import token_decode_attention_int8kv_int8pack_flash_decoding
from flashdecoding.int8_upperlower.int8kv_upperlower_flash_decoding import token_decode_attention_int8kv_upperlower_flash_decoding
from flashdecoding.int8_verify_upperlower.int8kv_verify_upperlower_flash_decoding import token_decode_attention_int8kv_verify_upperlower_flash_decoding

from quantize.quant_pack_int4toint32 import triton_int4toint32_quantize_and_pack_along_last_dim, triton_int4toint32_quantize_and_pack_along_penultimate_dim
from quantize.quant_pack_int4toint8 import triton_int4toint8_quantize_along_penultimate_dim_and_pack_along_last_dim
from quantize.quant_pack_int8toint8 import triton_int8toint8_quantize_along_penultimate_dim_and_pack_along_last_dim
from quantize.quant_pack_int8toint8_upperlower import triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim

from tests.test_utils import quantize_tensor_last_dim, quantize_tensor_penultimate_dim

group_size = 128
bit = 4
batch_size, num_heads, sequence_length, head_dim = 1, 32, 128 * 1024, 128
cache_len, residual_len = 128 * 1024, 272

num_runs = 20

def _init_simulate_kvcache(group_size, bit, B, nh, T, D):
    init_method = "random"
    dtype, device = torch.float16, "cuda"

    if init_method == "random":
        query_states = torch.rand((B, nh, 1, D), device=device, dtype=dtype)
        key_states = torch.rand((B, nh, T, D), device=device, dtype=dtype)
        value_states = torch.rand((B, nh, T, D), device=device, dtype=dtype)

    if init_method == "saved":
        query_states, key_states, value_states, dropout, softmax_scale, causal = \
            torch.load(f"debug_tensor/debug_llama_bs1_bit4_sl131072.pt", weights_only=True)

        # Hack the input # [B, T, nh, D]
        query_states, key_states, value_states = query_states[:B, :T, :nh, :head_dim], key_states[:B, :T, :nh, :head_dim].contiguous(), value_states[:B, :T, :nh, :head_dim].contiguous()
        
        # Transpose and permute to align the input of flashattention with our implementation
        query_states, key_states, value_states = query_states.transpose(1, 2).contiguous(), key_states.transpose(1, 2).contiguous(), value_states.transpose(1, 2).contiguous()

    return query_states, key_states, value_states

"""
Simulated MHA
"""
def _test_simulation_mha(query_states, key_states, value_states, B, nh, T, D):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(D)
    attn_output = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    output_no_quantize = torch.matmul(attn_output, value_states)

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

def _test_fp16_flash_decoding(query_states, key_states, value_states, cache_len):
    output = token_decode_attention_flash_decoding(query_states, key_states, value_states, qcache_len=cache_len)
    return output

"""
INT4 Pack to INT32 Flash Decoding
"""
def _init_int4toint32_quantize_kvcache(key_states, value_states, group_size, bit):
    # Real Quantize
    packquant_k, scale_k, min_k = triton_int4toint32_quantize_and_pack_along_penultimate_dim(key_states, group_size, bit)
    
    value_states = value_states.transpose(2, 3).contiguous()
    packquant_v, scale_v, min_v = triton_int4toint32_quantize_and_pack_along_penultimate_dim(value_states, group_size, bit)

    return packquant_k, scale_k, min_k, packquant_v, scale_v, min_v

def _test_int4toint32_flash_decoding(query_states, \
                                     packquant_k, scale_k, min_k, \
                                     packquant_v, scale_v, min_v, \
                                     group_size, bit,
                                     cache_len):
    
    output = token_decode_attention_int4kv_int32pack_flash_decoding(query_states, 
                                                          packquant_k, scale_k, min_k, 
                                                          packquant_v, scale_v, min_v,
                                                          bit, bit, group_size, 
                                                          qcache_len=cache_len)
    
    return output


"""
INT4 Pack to INT8 Flash Decoding
"""
def _init_int4toint8_quantize_kvcache(key_states, value_states, group_size, bit):
    # Real Quantize
    packquant_k, scale_k, min_k = triton_int4toint8_quantize_along_penultimate_dim_and_pack_along_last_dim(key_states, group_size, bit)
    
    value_states = value_states.transpose(2, 3).contiguous()
    packquant_v, scale_v, min_v = triton_int4toint8_quantize_along_penultimate_dim_and_pack_along_last_dim(value_states, group_size, bit)

    return packquant_k, scale_k, min_k, packquant_v, scale_v, min_v

def _test_int4toint8_flash_decoding(query_states, \
                                    packquant_k, scale_k, min_k, \
                                    packquant_v, scale_v, min_v, \
                                    group_size, bit, 
                                    full_k=None, full_v=None,
                                    cache_len=0, residual_len=0):
    
    output = token_decode_attention_int4kv_int8pack_flash_decoding(query_states, 
                                                          packquant_k, scale_k, min_k, 
                                                          packquant_v, scale_v, min_v,
                                                          bit, bit, group_size,
                                                          full_k, full_v,
                                                          qcache_len=cache_len, residual_len=residual_len)
    
    return output


"""
INT8 Pack to INT8 Flash Decoding
"""
def _init_int8toint8_quantize_kvcache(key_states, value_states, group_size, bit):
    # Real Quantize
    packquant_k, scale_k, min_k = triton_int8toint8_quantize_along_penultimate_dim_and_pack_along_last_dim(key_states, group_size, bit)
    
    value_states = value_states.transpose(2, 3).contiguous()
    packquant_v, scale_v, min_v = triton_int8toint8_quantize_along_penultimate_dim_and_pack_along_last_dim(value_states, group_size, bit)

    return packquant_k, scale_k, min_k, packquant_v, scale_v, min_v

def _test_int8toint8_flash_decoding(query_states, \
                                    packquant_k, scale_k, min_k, \
                                    packquant_v, scale_v, min_v, \
                                    group_size, bit, 
                                    full_k=None, full_v=None,
                                    cache_len=0, residual_len=0):
    
    output = token_decode_attention_int8kv_int8pack_flash_decoding(query_states, 
                                                          packquant_k, scale_k, min_k, 
                                                          packquant_v, scale_v, min_v,
                                                          bit, bit, group_size,
                                                          full_k, full_v,
                                                          qcache_len=cache_len, residual_len=residual_len)
    
    return output


"""
INT8 Packed with Upper 4-bit and Lower 4-bit, they are all packed with INT8. Flash Decoding.
"""
def _init_int8_upperlower_quantize_kvcache(key_states, value_states, group_size, bit):
    # Real Quantize
    packquant_k_upper, packquant_k_lower, scale_k, min_k = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(key_states, group_size, bit)
    
    value_states = value_states.transpose(2, 3).contiguous()
    packquant_v_upper, packquant_v_lower, scale_v, min_v = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(value_states, group_size, bit)

    return packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v

def _test_int8_upperlower_flash_decoding(query_states, \
                                         packquant_k_upper, packquant_k_lower, scale_k, min_k, \
                                         packquant_v_upper, packquant_v_lower, scale_v, min_v, \
                                         group_size, bit, \
                                         full_k=None, full_v=None,
                                         cache_len=0, residual_len=0):
    
    cache_len = torch.tensor(cache_len, dtype=torch.int32).to(query_states.device)
    residual_len = torch.tensor(residual_len, dtype=torch.int32).to(query_states.device)
    
    output = token_decode_attention_int8kv_upperlower_flash_decoding(query_states, 
                                                          packquant_k_upper, packquant_k_lower, scale_k, min_k, 
                                                          packquant_v_upper, packquant_v_lower, scale_v, min_v,
                                                          bit, bit, group_size, \
                                                          full_k, full_v,
                                                          max_seq_length=cache_len, max_residual_len=residual_len,
                                                          qcache_len=cache_len, residual_len=residual_len)
    
    return output


"""
INT8 Packed with Upper 4-bit and Lower 4-bit, they are all packed with INT8. Flash Decoding. Q Sequence Length > 1
"""
def _init_int8_verify_upperlower_quantize_kvcache(key_states, value_states, group_size, bit):
    # Real Quantize
    packquant_k_upper, packquant_k_lower, scale_k, min_k = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(key_states, group_size, bit)
    
    value_states = value_states.transpose(2, 3).contiguous()
    packquant_v_upper, packquant_v_lower, scale_v, min_v = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(value_states, group_size, bit)

    return packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v

def _test_int8_verify_upperlower_flash_decoding(query_states, \
                                         packquant_k_upper, packquant_k_lower, scale_k, min_k, \
                                         packquant_v_upper, packquant_v_lower, scale_v, min_v, \
                                         group_size, bit, \
                                         full_k=None, full_v=None,
                                         cache_len=0, residual_len=0):
    
    cache_len = torch.tensor(cache_len, dtype=torch.int32).to(query_states.device)
    residual_len = torch.tensor(residual_len, dtype=torch.int32).to(query_states.device)
    
    output = token_decode_attention_int8kv_verify_upperlower_flash_decoding(query_states, 
                                                          packquant_k_upper, packquant_k_lower, scale_k, min_k, 
                                                          packquant_v_upper, packquant_v_lower, scale_v, min_v,
                                                          bit, bit, group_size, \
                                                          full_k, full_v,
                                                          max_seq_length=cache_len, max_residual_len=residual_len,
                                                          qcache_len=cache_len, residual_len=residual_len)
    
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
        query_states, key_states, value_states = _init_simulate_kvcache(group_size, bit, batch_size, num_heads, sequence_length, head_dim)
        simulated_output = _test_simulation_mha(query_states, key_states, value_states, \
                                                batch_size, num_heads, sequence_length, head_dim)
        
        _value_states = value_states.transpose(2, 3).contiguous()
        fp16_output = _test_fp16_flash_decoding(query_states, key_states, _value_states, cache_len=cache_len)
        median_time = measure_cuda_timing(
            _test_fp16_flash_decoding,
            20, 
            query_states, key_states, _value_states, cache_len=cache_len
        )
        print(f"Median execution time of Flash Decoding FP16: {median_time} ms")


        packquant_k, scale_k, min_k, packquant_v, scale_v, min_v = _init_int4toint32_quantize_kvcache(key_states, value_states, group_size, bit)
        int4toint32_output = _test_int4toint32_flash_decoding(query_states, \
                                                              packquant_k, scale_k, min_k, \
                                                              packquant_v, scale_v, min_v, \
                                                              group_size, bit, cache_len=cache_len)


        packquant_k, scale_k, min_k, packquant_v, scale_v, min_v = _init_int4toint8_quantize_kvcache(key_states, value_states, group_size, bit)
        int4toint8_output = _test_int4toint8_flash_decoding(query_states, \
                                                            packquant_k, scale_k, min_k, \
                                                            packquant_v, scale_v, min_v, \
                                                            group_size, bit,
                                                            cache_len=cache_len, residual_len=residual_len)
        median_time = measure_cuda_timing(
            _test_int4toint8_flash_decoding,
            20, 
            query_states, packquant_k, scale_k, min_k, packquant_v, scale_v, min_v, group_size, bit, cache_len=cache_len, residual_len=residual_len
        )
        print(f"Median execution time of Flash Decoding INT4 to INT8: {median_time} ms")


        packquant_k, scale_k, min_k, packquant_v, scale_v, min_v = _init_int8toint8_quantize_kvcache(key_states, value_states, group_size, 2 * bit)
        int8toint8_output = _test_int8toint8_flash_decoding(query_states, \
                                                            packquant_k, scale_k, min_k, \
                                                            packquant_v, scale_v, min_v, \
                                                            group_size, 2 * bit, 
                                                            cache_len=cache_len, residual_len=residual_len)

        median_time = measure_cuda_timing(
            _test_int8toint8_flash_decoding,
            20, 
            query_states, packquant_k, scale_k, min_k, packquant_v, scale_v, min_v, group_size, 2 * bit, cache_len=cache_len, residual_len=residual_len
        )
        print(f"Median execution time of Flash Decoding INT8 to INT8: {median_time} ms")


        packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v = _init_int8_upperlower_quantize_kvcache(key_states, value_states, group_size, bit)
        int8_upperlower_output = _test_int8_upperlower_flash_decoding(query_states, \
                                                                      packquant_k_upper, packquant_k_lower, scale_k, min_k, \
                                                                      packquant_v_upper, packquant_v_lower, scale_v, min_v, \
                                                                      group_size, bit, 
                                                                      cache_len=cache_len, residual_len=residual_len)
        median_time = measure_cuda_timing(
            _test_int8_upperlower_flash_decoding,
            20, 
            query_states, packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v, 
            group_size, bit, cache_len=cache_len, residual_len=residual_len
        )
        print(f"Median execution time of Flash Decoding INT8 UpperLower: {median_time} ms")


        print("Benchmark without Full Part")
        print(f"FP16 FlashDecoding Error: {(simulated_output - fp16_output).norm()}")
        print(f"INT4 Pack in INT32 Error: {(simulated_output - int4toint32_output).norm()}")
        print(f"INT4 Pack in INT8  Error: {(simulated_output - int4toint8_output).norm()}")
        print(f"INT8 Pack in INT8  Error: {(simulated_output - int8toint8_output).norm()}")
        print(f"INT8 UpperLower    Error: {(simulated_output - int8_upperlower_output).norm()}")
        print("\n")

        _, full_key_states, full_value_states = _init_simulate_kvcache(group_size, bit, batch_size, num_heads, residual_len, head_dim)
        simulated_output = _test_simulation_mha(query_states, torch.cat([key_states, full_key_states], dim=2), torch.cat([value_states, full_value_states], dim=2), \
                                                batch_size, num_heads, sequence_length, head_dim)

        packquant_k, scale_k, min_k, packquant_v, scale_v, min_v = _init_int4toint8_quantize_kvcache(key_states, value_states, group_size, bit)
        int4toint8_output = _test_int4toint8_flash_decoding(query_states, \
                                                            packquant_k, scale_k, min_k, \
                                                            packquant_v, scale_v, min_v, \
                                                            group_size, bit, \
                                                            full_key_states, full_value_states, 
                                                            cache_len=cache_len, residual_len=residual_len)

        median_time = measure_cuda_timing(
            _test_int4toint8_flash_decoding,
            20, 
            query_states, packquant_k, scale_k, min_k, packquant_v, scale_v, min_v, 
            group_size, bit, full_key_states, full_value_states, cache_len=cache_len, residual_len=residual_len
        )
        print(f"Median execution time of Flash Decoding INT4 to INT8 with Full part: {median_time} ms")

        packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v = _init_int8_upperlower_quantize_kvcache(key_states, value_states, group_size, bit)
        int8_upperlower_output = _test_int8_upperlower_flash_decoding(query_states, \
                                                                      packquant_k_upper, packquant_k_lower, scale_k, min_k, \
                                                                      packquant_v_upper, packquant_v_lower, scale_v, min_v, \
                                                                      group_size, bit, \
                                                                      full_key_states, full_value_states, 
                                                                      cache_len=cache_len, residual_len=residual_len)
        median_time = measure_cuda_timing(
            _test_int8_upperlower_flash_decoding,
            20, 
            query_states, packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v, \
            group_size, bit, full_key_states, full_value_states, cache_len=cache_len, residual_len=residual_len
        )
        print(f"Median execution time of Flash Decoding INT8 UpperLower with Full part: {median_time} ms")


        print("\nBenchmarking with Full Part")
        print(f"INT4 Pack in INT8 with Full Part  Error: {(simulated_output - int4toint8_output).norm()}")
        print(f"INT8 UpperLower with Full Part    Error: {(simulated_output - int8_upperlower_output).norm()}")
        print("\n")


        """ Query Sequence Length > 1 """
        q_seq_len = 6
        verify_query_states = query_states.repeat(1, 1, q_seq_len, 1)

        packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v = _init_int8_verify_upperlower_quantize_kvcache(key_states, value_states, group_size, bit)
        int8_verify_upperlower_query_output = _test_int8_verify_upperlower_flash_decoding(verify_query_states, \
                                                                      packquant_k_upper, packquant_k_lower, scale_k, min_k, \
                                                                      packquant_v_upper, packquant_v_lower, scale_v, min_v, \
                                                                      group_size, bit, 
                                                                      cache_len=cache_len, residual_len=residual_len)

        verify_query_simulated_output = _test_simulation_verify_query_mha(verify_query_states, key_states, value_states, \
                                                batch_size, num_heads, sequence_length, head_dim)

        median_time = measure_cuda_timing(
            _test_int8_verify_upperlower_flash_decoding,
            20, 
            verify_query_states, packquant_k_upper, packquant_k_lower, scale_k, min_k, \
            packquant_v_upper, packquant_v_lower, scale_v, min_v, \
            group_size, bit, cache_len=cache_len, residual_len=residual_len
        )

        print(f"Median execution time of Flash Decoding INT8 UpperLower Without Full part, Query Sequence Length = {q_seq_len}: {median_time} ms")

        q_seq_len = 6
        verify_query_states = query_states.repeat(1, 1, q_seq_len, 1)

        packquant_k_upper, packquant_k_lower, scale_k, min_k, packquant_v_upper, packquant_v_lower, scale_v, min_v = _init_int8_verify_upperlower_quantize_kvcache(key_states, value_states, group_size, bit)
        int8_verify_upperlower_query_output = _test_int8_verify_upperlower_flash_decoding(verify_query_states, \
                                                                      packquant_k_upper, packquant_k_lower, scale_k, min_k, \
                                                                      packquant_v_upper, packquant_v_lower, scale_v, min_v, \
                                                                      group_size, bit,
                                                                      full_key_states, full_value_states, 
                                                                      cache_len=cache_len, residual_len=residual_len)

        verify_query_simulated_output = _test_simulation_verify_query_mha(verify_query_states, key_states, value_states, \
                                                batch_size, num_heads, sequence_length, head_dim)

        median_time = measure_cuda_timing(
            _test_int8_verify_upperlower_flash_decoding,
            20, 
            verify_query_states, packquant_k_upper, packquant_k_lower, scale_k, min_k, \
            packquant_v_upper, packquant_v_lower, scale_v, min_v, \
            group_size, bit, full_key_states, full_value_states, 
            cache_len=cache_len, residual_len=residual_len
        )

        print(f"Median execution time of Flash Decoding INT8 UpperLower with    Full part, Query Sequence Length = {q_seq_len}: {median_time} ms")

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False, linewidth=200)
    unittest.main()