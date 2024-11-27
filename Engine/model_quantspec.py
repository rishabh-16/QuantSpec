from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
import math 
from QuantSpec_magidec.kernels.quantize.quant_pack_int8toint8_upperlower import triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim
from QuantSpec_magidec.kernels.flashdecoding.int8_verify_upperlower.int8kv_verify_upperlower_flash_decoding import token_decode_attention_int8kv_verify_upperlower_flash_decoding
from QuantSpec_magidec.kernels.flashdecoding.int8_upperlower.int8kv_upperlower_flash_decoding import token_decode_attention_int8kv_upperlower_flash_decoding

# from QuantSpec_magidec.kernels.flashdecoding.int8_failed_verify_upperlower.int8kv_verify_upperlower_flash_decoding import token_decode_attention_int8kv_verify_upperlower_flash_decoding

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    scaling_factor:float = 1.0
    # llama 3.1 with high_freq_factor and low_freq_factor
    low_freq_factor: int = None # added new
    high_freq_factor: int = None  # added new
    original_max_position_embeddings: int = None   # added new

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]
        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
        print(config)
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "llama-2-7b": dict(block_size=4096, n_layer=32, n_head=32, dim=4096),
    'llama-2-7b-32k': dict(block_size=32768, n_layer=32, dim= 4096, vocab_size=32000, scaling_factor=8),
    "llama-2-13b": dict(block_size=4096, n_layer=40, n_head=40, dim=5120),
    "llama-2-70b": dict(block_size=4096, n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "68m": dict(block_size=2048, n_layer=2, n_head=12, n_local_heads=12, dim=768, intermediate_size=3072, vocab_size=32000),
    "tinyllama": dict(block_size =2048, n_layer=22, n_head=32, n_local_heads=4, dim=2048, intermediate_size=5632, vocab_size=32000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000.0, scaling_factor=8, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16, **cache_kwargs):
        super().__init__()
        qtype = torch.int8
        self.max_seq_length = max_seq_length
        self.residual_len = cache_kwargs.get("residual_len", 128)
        self.group_size = cache_kwargs.get("group_size", 128)
        self.k_bits = cache_kwargs.get("k_bits", 4)
        self.v_bits = cache_kwargs.get("v_bits", 4)

        residual_cache_shape = (max_batch_size, n_heads, 2*self.residual_len+1, head_dim)
        qkey_cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim//(8//self.k_bits))
        keymin_cache_shape = (max_batch_size, n_heads, max_seq_length//self.group_size, head_dim)
        keyscale_cache_shape = (max_batch_size, n_heads, max_seq_length//self.group_size, head_dim)

        qval_trans_cache_shape = (max_batch_size, n_heads, head_dim, max_seq_length//(8//self.v_bits))
        valmin_cache_shape = (max_batch_size, n_heads, head_dim//self.group_size, max_seq_length)
        valscale_cache_shape = (max_batch_size, n_heads, head_dim//self.group_size, max_seq_length)

        self.register_buffer('k_cache', torch.zeros(residual_cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(residual_cache_shape, dtype=dtype))
        self.register_buffer('qk_cache_ubits', torch.zeros(qkey_cache_shape, dtype=qtype))
        self.register_buffer('qk_cache_lbits', torch.zeros(qkey_cache_shape, dtype=qtype))
        self.register_buffer('kmin_cache', torch.zeros(keymin_cache_shape, dtype=dtype))
        self.register_buffer('kscale_cache', torch.zeros(keyscale_cache_shape, dtype=dtype))
        self.register_buffer('qval_trans_cache_ubits', torch.zeros(qval_trans_cache_shape, dtype=qtype))
        self.register_buffer('qval_trans_cache_lbits', torch.zeros(qval_trans_cache_shape, dtype=qtype))
        self.register_buffer('valmin_cache', torch.zeros(valmin_cache_shape, dtype=dtype))
        self.register_buffer('valscale_cache', torch.zeros(valscale_cache_shape, dtype=dtype))
        self.register_buffer('batch_indices',torch.arange(max_batch_size).unsqueeze(1))


    def prefill_update(self, k_cache, v_cache):
        output = k_cache, v_cache

        prefill_len = k_cache.shape[2]
        assert prefill_len == v_cache.shape[2], "k and v must have the same length in prefill"
        if prefill_len > 2*self.residual_len:
            assert self.residual_len % self.group_size == 0, "Residual length must be divisible by group size"
            to_quant_len = prefill_len-self.residual_len-(prefill_len%self.group_size)
            if prefill_len-to_quant_len > 0:
                self.k_cache[:, :, :prefill_len-to_quant_len].copy_(k_cache[:, :, to_quant_len:].clone())
                self.v_cache[:, :, :prefill_len-to_quant_len].copy_(v_cache[:, :, to_quant_len:].clone())

            qk_cache_ubits, qk_cache_lbits, kscale_cache, kmin_cache = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(k_cache[:, :, :to_quant_len].clone().contiguous(), self.group_size, self.k_bits)
            self.qk_cache_ubits[:, :, :to_quant_len].copy_(qk_cache_ubits)
            self.qk_cache_lbits[:, :, :to_quant_len].copy_(qk_cache_lbits)
            self.kmin_cache[:, :, :to_quant_len//self.group_size].copy_(kmin_cache)
            self.kscale_cache[:, :, :to_quant_len//self.group_size].copy_(kscale_cache)
            
            qval_trans_cache_ubits, qval_trans_cache_lbits, valscale_cache, valmin_cache = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(v_cache[:, :, :to_quant_len].clone().transpose(2,3).contiguous(), self.group_size, self.v_bits)
            self.qval_trans_cache_ubits[:, :, :, :to_quant_len//(8//self.v_bits)].copy_(qval_trans_cache_ubits)
            self.qval_trans_cache_lbits[:, :, :, :to_quant_len//(8//self.v_bits)].copy_(qval_trans_cache_lbits)
            self.valmin_cache[:, :, :, :to_quant_len].copy_(valmin_cache)
            self.valscale_cache[:, :, :, :to_quant_len].copy_(valscale_cache)
        else:
            self.k_cache[:, :, :prefill_len].copy_(k_cache)
            self.v_cache[:, :, :prefill_len].copy_(v_cache)
        return output
    
    def update(self, k_cache, v_cache, cache_seqlens, qcache_seqlens):

        # TODO: make this more efficient
        # residual_lens = cache_seqlens - qcache_seqlens
        # bsz = k_cache.shape[0]
        # assert bsz == 1, "Batch size > 1 not supported yet"
        # for b in range(bsz):
        #     start_idx = residual_lens[b]
        #     end_idx = residual_lens[b] + k_cache.shape[2]
        #     self.k_cache[b, :, start_idx:end_idx, :].copy_(k_cache[b])
        #     self.v_cache[b, :, start_idx:end_idx, :].copy_(v_cache[b])

        bsz, num_heads, update_len, head_dim = k_cache.shape
        seq_len = self.k_cache.shape[2]

        residual_lens = cache_seqlens - qcache_seqlens
        
        # Create index tensor for scatter
        indices = torch.arange(update_len, device=k_cache.device)
        indices = indices.view(1, 1, -1, 1)  # [1, 1, update_len, 1]
        indices = indices + residual_lens.view(-1, 1, 1, 1)  # [bsz, 1, update_len, 1]
        indices = indices.expand(bsz, num_heads, update_len, head_dim)
        
        # Perform scatter operations
        self.k_cache.scatter_(dim=2, index=indices, src=k_cache)
        self.v_cache.scatter_(dim=2, index=indices, src=v_cache)

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.qk_cache_ubits.zero_()
        self.qk_cache_lbits.zero_()
        self.kmin_cache.zero_()
        self.kscale_cache.zero_()
        self.qval_trans_cache_ubits.zero_()
        self.qval_trans_cache_lbits.zero_()
        self.valmin_cache.zero_()
        self.valscale_cache.zero_()

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, **cache_kwargs):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for i, b in enumerate(self.layers):
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype, **cache_kwargs)
            b.attention.layer_idx = i

        if (self.config.high_freq_factor is not None) and (self.config.low_freq_factor is not None):
            self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base,dtype,
                                                  # new params
                                                  self.config.scaling_factor, self.config.low_freq_factor, self.config.high_freq_factor, self.config.original_max_position_embeddings)
        else:
            self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base,dtype,
                                                  # new params
                                                  self.config.scaling_factor)

    def forward(self, idx: Tensor, input_pos: Optional[Tensor], cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, freqs_cis, cache_seqlens, qcache_seqlens)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    def prefill(self, idx: Tensor, input_pos: Optional[Tensor], cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.prefill(x, freqs_cis, cache_seqlens, qcache_seqlens)
        x = self.norm(x)
        logits = self.output(x)
        return logits
    
    def draft_forward(self, idx: Tensor, input_pos: Optional[Tensor], cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"

        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer.draft_forward(x, freqs_cis, cache_seqlens, qcache_seqlens)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, cache_seqlens, qcache_seqlens)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def prefill(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        h = x + self.attention.prefill(self.attention_norm(x), freqs_cis, cache_seqlens, qcache_seqlens)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def draft_forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        h = x + self.attention.draft_forward(self.attention_norm(x), freqs_cis, cache_seqlens, qcache_seqlens)
        out = h + self.feed_forward.draft_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.process_group = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

        if self.n_head == self.n_local_heads:
            self._attn = torch.ops.mylib.custom_func
        else:
            self._attn = torch.ops.mylib.gqa_custom

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])
    
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        if hidden_states is None:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


    def forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        # TODO: check transpose, might affect rotary emb
        q = apply_rotary_emb(q.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis).transpose(1,2)
        k = apply_rotary_emb(k.view(bsz, seqlen, self.n_local_heads, self.head_dim), freqs_cis).transpose(1,2)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1,2)

        self.kv_cache.update(k, v, cache_seqlens, qcache_seqlens)
        cache_seqlens = cache_seqlens + k.shape[-2]
        
        repeat_factor = self.n_head // self.n_local_heads

        y = token_decode_attention_int8kv_verify_upperlower_flash_decoding(
            q=q,
            cache_quant_k_upper=self.repeat_kv(self.kv_cache.qk_cache_ubits, repeat_factor),
            cache_quant_k_lower=self.repeat_kv(self.kv_cache.qk_cache_lbits, repeat_factor), 
            cache_scale_k=self.repeat_kv(self.kv_cache.kscale_cache, repeat_factor),
            cache_min_k=self.repeat_kv(self.kv_cache.kmin_cache, repeat_factor),
            cache_quant_v_upper=self.repeat_kv(self.kv_cache.qval_trans_cache_ubits, repeat_factor),
            cache_quant_v_lower=self.repeat_kv(self.kv_cache.qval_trans_cache_lbits, repeat_factor),
            cache_scale_v=self.repeat_kv(self.kv_cache.valscale_cache, repeat_factor),
            cache_min_v=self.repeat_kv(self.kv_cache.valmin_cache, repeat_factor),
            kbit=self.kv_cache.k_bits,
            vbit=self.kv_cache.v_bits,
            group_size=self.kv_cache.group_size,
            full_k=self.repeat_kv(self.kv_cache.k_cache, repeat_factor),
            full_v=self.repeat_kv(self.kv_cache.v_cache, repeat_factor),
            out=None,
            alloc_tensor_func=torch.zeros,
            precision=8,
            max_seq_length=self.kv_cache.max_seq_length,
            max_residual_len=2 * self.kv_cache.residual_len + 1,
            qcache_len=qcache_seqlens[0],
            residual_len=cache_seqlens[0] - qcache_seqlens[0],
        )

        y = y.transpose(1, 2).reshape(bsz, seqlen, self.dim).contiguous()

        
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y)

        return y

    def prefill(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        
        k, v = self.kv_cache.prefill_update(k.transpose(1,2), v.transpose(1,2))

        # for prefill, use original impl

        y = torch.ops.mylib.custom_func_2(q, k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous())
        y = y.contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y)
        return y
    
    def draft_forward(self, x: Tensor, freqs_cis: Tensor, cache_seqlens: Tensor, qcache_seqlens: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        # TODO: check transpose
        q = apply_rotary_emb(q.view(bsz, seqlen, self.n_head, self.head_dim), freqs_cis).transpose(1,2)
        k = apply_rotary_emb(k.view(bsz, seqlen, self.n_local_heads, self.head_dim), freqs_cis).transpose(1,2)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1,2)
        
        # if self.kv_cache is not None:
        #     k, v = self.kv_cache.update(cache_seqlens, k, v)

        self.kv_cache.update(k, v, cache_seqlens, qcache_seqlens)
        cache_seqlens = cache_seqlens + k.shape[-2]

        repeat_factor = self.n_head // self.n_local_heads

        # y = torch.ops.mylib.custom_func_2(q, k, v)

        y = token_decode_attention_int8kv_upperlower_flash_decoding(
            q=q,
            cache_quant_k_upper=self.repeat_kv(self.kv_cache.qk_cache_ubits, repeat_factor),
            cache_quant_k_lower=self.repeat_kv(self.kv_cache.qk_cache_lbits, repeat_factor), 
            cache_scale_k=self.repeat_kv(self.kv_cache.kscale_cache, repeat_factor),
            cache_min_k=self.repeat_kv(self.kv_cache.kmin_cache, repeat_factor),
            cache_quant_v_upper=self.repeat_kv(self.kv_cache.qval_trans_cache_ubits, repeat_factor),
            cache_quant_v_lower=self.repeat_kv(self.kv_cache.qval_trans_cache_lbits, repeat_factor),
            cache_scale_v=self.repeat_kv(self.kv_cache.valscale_cache, repeat_factor),
            cache_min_v=self.repeat_kv(self.kv_cache.valmin_cache, repeat_factor),
            kbit=self.kv_cache.k_bits,
            vbit=self.kv_cache.v_bits,
            group_size=self.kv_cache.group_size,
            full_k=self.repeat_kv(self.kv_cache.k_cache, repeat_factor),
            full_v=self.repeat_kv(self.kv_cache.v_cache, repeat_factor),
            out=None,
            alloc_tensor_func=torch.zeros,
            precision=8,
            max_seq_length=self.kv_cache.max_seq_length,
            max_residual_len=2 * self.kv_cache.residual_len + 1,
            qcache_len=qcache_seqlens[0],
            residual_len=cache_seqlens[0] - qcache_seqlens[0],
        )
        
        # y = token_decode_attention_int8kv_upperlower_flash_decoding(
        #     q=q,
        #     cache_quant_k_upper=self.repeat_kv(self.kv_cache.qk_cache_ubits[:, :, :qcache_seqlens], repeat_factor),
        #     cache_quant_k_lower=self.repeat_kv(self.kv_cache.qk_cache_lbits[:, :, :qcache_seqlens], repeat_factor), 
        #     cache_scale_k=self.repeat_kv(self.kv_cache.kscale_cache[:, :, :qcache_seqlens//self.kv_cache.group_size], repeat_factor),
        #     cache_min_k=self.repeat_kv(self.kv_cache.kmin_cache[:, :, :qcache_seqlens//self.kv_cache.group_size], repeat_factor),
        #     cache_quant_v_upper=self.repeat_kv(self.kv_cache.qval_trans_cache_ubits[:, :, :, :qcache_seqlens//(8//self.kv_cache.v_bits)], repeat_factor),
        #     cache_quant_v_lower=self.repeat_kv(self.kv_cache.qval_trans_cache_lbits[:, :, :, :qcache_seqlens//(8//self.kv_cache.v_bits)], repeat_factor),
        #     cache_scale_v=self.repeat_kv(self.kv_cache.valscale_cache[:, :, :, :qcache_seqlens], repeat_factor),
        #     cache_min_v=self.repeat_kv(self.kv_cache.valmin_cache[:, :, :, :qcache_seqlens], repeat_factor),
        #     kbit=self.kv_cache.k_bits,
        #     vbit=self.kv_cache.v_bits,
        #     group_size=self.kv_cache.group_size,
        #     full_k=self.repeat_kv(self.kv_cache.k_cache[:, :, :cache_seqlens], repeat_factor),
        #     full_v=self.repeat_kv(self.kv_cache.v_cache[:, :, :cache_seqlens], repeat_factor),
        #     out=None,
        #     alloc_tensor_func=torch.zeros,
        #     precision=8,
        # )

        y = y.transpose(1, 2).reshape(bsz, seqlen, self.dim).contiguous()

        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y)

        return y

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.process_group = None

    def forward(self, x: Tensor) -> Tensor:
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.process_group != None:
            dist.all_reduce(y)
        return y
    
    def draft_forward(self, x: Tensor) -> Tensor:
        y = self.w2(F.silu(self.w1(x)) * self.w3(x))
        if self.process_group != None:
            dist.all_reduce(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def _compute_llama3_parameters(inv_freq, old_context_len=8192, scaling_factor=8,low_freq_factor=1,high_freq_factor=4):
    """
    To be used for llama 3.1 models
        - borrowing the logic from: https://github.com/huggingface/transformers/blob/c85510f958e6955d88ea1bafb4f320074bfbd0c1/src/transformers/modeling_rope_utils.py
        - source: _compute_llama3_parameters in modeling_rope_utils.py
    """
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scaling_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scaling_factor + smooth * freq)
    inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
    return inv_freq

# def precompute_freqs_cis(
#     seq_len: int, n_elem: int, base: int = 10000,
#     dtype: torch.dtype = torch.bfloat16,
#     scaling_factor = 1
# ) -> Tensor:
#     freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
#     freqs /= scaling_factor
#     t = torch.arange(seq_len, device=freqs.device, dtype=freqs.dtype)
#     # t /=scaling_factor
#     freqs = torch.outer(t, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
#     cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
#     return cache.to(dtype=dtype)

def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    scaling_factor: float = 1.0, # added new 
    low_freq_factor: int = None, # added new
    high_freq_factor: int = None, # added new
    original_max_position_embeddings: int = None, # added new
) -> Tensor:
    print(f"target: seq_len: {seq_len}, n_elem: {n_elem}, base: {base}, dtype: {dtype}, scaling_factor: {scaling_factor}, low_freq_factor: {low_freq_factor}, high_freq_factor: {high_freq_factor}, original_max_position_embeddings: {original_max_position_embeddings}"
          )
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    
    if (low_freq_factor is not None) and (high_freq_factor is not None):
        freqs = _compute_llama3_parameters(freqs, original_max_position_embeddings, scaling_factor, low_freq_factor,high_freq_factor)
    else:
        freqs /= scaling_factor
    t = torch.arange(seq_len, device=freqs.device, dtype=freqs.dtype)
    # t /=scaling_factor
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)



def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(x.shape[0], xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)