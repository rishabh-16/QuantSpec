import math
import torch
import torch.nn as nn

def token_decode_attention_flash_decoding(
    q, cache_k, cache_v, out=None, alloc_tensor_func=torch.zeros, qcache_len=0
):
    """
    q : torch.Tensor
        The query tensor of shape (batch_size, num_heads, 1, head_dim). It represents 
        the query for the current token at the specific decoding step.
    
    cache_k : torch.Tensor
        The cached key tensor of shape (batch_size, num_heads, seq_length, head_dim). 

    cache_v : torch.Tensor
        The cached value tensor of shape (batch_size, num_heads, head_dim, seq_length). 
        It is transposed to meet the requirement of decoding
    
    out : torch.Tensor, optional
        Output tensor to store the result of shape (batch_size, num_heads, 1, head_dim).
    """
    BLOCK_SEQ = 512
    batch_size = q.shape[0]
    q_head_num, head_dim = q.shape[1], q.shape[-1]
    
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from .flash_decoding_stage1 import flash_decode_stage1
    from .flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(tuple(q.shape), dtype=q.dtype, device=q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, qcache_len // BLOCK_SEQ + 1, head_dim], dtype=torch.float32, device=q.device
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, qcache_len // BLOCK_SEQ + 1], dtype=torch.float32, device=q.device
    )

    flash_decode_stage1(
        q.view(calcu_shape1),
        cache_k,
        cache_v,
        mid_o,
        mid_o_logexpsum,
        qcache_len,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, o_tensor.view(calcu_shape1), qcache_len, BLOCK_SEQ)
    return o_tensor
