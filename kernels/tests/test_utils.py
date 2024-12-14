import torch

def quantize_tensor_last_dim(inp, group_size, bits): # Along the last dimension
    num_groups = inp.size(-1) // group_size
    
    inp = inp.reshape(*inp.shape[:-1], num_groups, group_size)
    maxval = torch.max(inp, dim=-1).values
    minval = torch.min(inp, dim=-1).values

    scale = (maxval - minval)/(2**bits - 1)
    q_inp = inp - minval.unsqueeze(-1)
    q_inp = q_inp / scale.unsqueeze(-1)

    q_inp = torch.clip(q_inp, min=0, max=2**bits-1).round().to(torch.int32)

    rq_inp = (q_inp * scale.unsqueeze(-1)) + minval.unsqueeze(-1)

    return rq_inp.reshape(*inp.shape[:-2], -1), q_inp.reshape(*inp.shape[:-2], -1), scale, minval

def quantize_tensor_penultimate_dim(inp, group_size, bit):
    rq_inp, quant_inp, scale_inp, min_inp = quantize_tensor_last_dim(inp.transpose(2, 3).contiguous(), group_size, bit)
    rq_inp, quant_inp, scale_inp, min_inp = \
        rq_inp.transpose(2, 3).contiguous(), quant_inp.transpose(2, 3).contiguous(), \
        scale_inp.transpose(2, 3).contiguous(), min_inp.transpose(2, 3).contiguous()
    return rq_inp, quant_inp, scale_inp, min_inp
