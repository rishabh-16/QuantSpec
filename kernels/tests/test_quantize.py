import torch
import unittest
from quantize.quant_pack_int8toint8_upperlower import triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim

# Simulated
def quantize_and_pack_along_penultimate_dim(inp, group_size, bits):
        num_groups = inp.size(-2) // group_size
        inp = inp.reshape(*inp.shape[:-2], num_groups, group_size, -1)
        maxval = torch.max(inp, dim=-2).values
        minval = torch.min(inp, dim=-2).values
        scale = (maxval - minval)/(2**(bits) - 1)
        ubits = inp - minval.unsqueeze(-2)
        ubits = ubits / scale.unsqueeze(-2)
        q_ubits = torch.clip(ubits, min=0, max=2**(bits)-1).round().to(torch.int32)
        q_error = ubits - q_ubits
        lbits = q_error*(2**(bits))
        q_lbits = torch.clip(lbits, min=-2**(bits - 1), max=2**(bits - 1)-1).round().to(torch.int32)
        return q_ubits.reshape(*inp.shape[:-3], -1, inp.shape[-1]), q_lbits.reshape(*inp.shape[:-3], -1, inp.shape[-1]), scale, minval


class TestQuantize(unittest.TestCase):
    def test_quantize_int8_upperlower(self):
        # x_tensor = torch.randn((1, 32, 2048, 128), device="cuda", dtype=torch.float16)
        x_tensor = torch.load("debug_tensor.pt")
        group_size, bits = 128, 4
        q_ubits, q_lbits, scale, min = quantize_and_pack_along_penultimate_dim(x_tensor, group_size, bits)

        code_upper, code_lower, code_scale, code_min = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(x_tensor, group_size, bits)

        x_tensor_1 = x_tensor.to("cuda:1")
        code_upper_1, code_lower_1, code_scale_1, code_min_1 = triton_int8toint8_upperlower_quantize_along_penultimate_dim_and_pack_along_last_dim(x_tensor_1, group_size, bits)

        import IPython
        IPython.embed()
          

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(linewidth=200)
    unittest.main()