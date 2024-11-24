# QuantSpec_magidec

Quantized Speculative Decoding

## Current Status

### Working Features
- Basic quantization and dequantization functionality
- Core decoding operations
- Unit tests passing for most components

### Known Issues

1. **Verification Phase Output Mismatch**
   - The verification phase is currently producing incorrect output
   - This could be due to either:
     - Kernel implementation issues
     - Core logic issues in the verification code
   - Under investigation

2. **CUDA Graph Limitations** 
   - CUDA graph implementation is encountering errors due to data-dependent operations in kernel
   - Basic CUDA graph functionality works but needs fixes for full compatibility

### TODOs: Haocheng

In priority order:
- [ ] kernels should take cache_len and residual_len as tensors not as int (if this is not possible let Rishabh know)
- [ ] Once the above step is done, run ```tests/run.sh 0``` to run the code with cuda graph and fix the kernels side errors.
- [ ] Check the verification kernel to see if it is correct. Currently it is giving wrong output for me. 


### Upcoming Features

1. **Weight Quantization**
   - Support for quantizing model weights
   - Integration with existing cache quantization

2. **Mixed Precision Support**
   - Implementation of mixed precision operations
   - Flexible precision selection for different components

## Contributing

If you'd like to help resolve any of the known issues or implement upcoming features, please feel free to submit a PR.

## License

[Add license information]
