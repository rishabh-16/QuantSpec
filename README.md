# QuantSpec_magidec

Quantized Speculative Decoding

## Installation

### Environment Set Up
``` bash
conda create -n quantspec python=3.11
conda activate quantspec
pip install torch --index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge gxx=13.2.0
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e marlin/
```

