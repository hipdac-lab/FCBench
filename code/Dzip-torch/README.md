# Dzip
## improved general-purpose lossless compression based on novel neural network modeling
## Description
DZip is a general lossless compressor for sequential data which uses NN-based modelling combined with arithmetic coding. We refer to the NN-based model as the "combined model", as it is composed of a bootstrap model and a supporter model. The bootstrap model is trained prior to compression on the data to be compressed, and the resulting model parameters (weights) are stored as part of the compressed output (after being losslessly compressed with BSC). The combined model is adaptively trained (bootstrap model parameters are fixed) while compressing the data, and hence its parameters do not need to be stored as part of the compressed output.

The article can be accessed [here](https://arxiv.org/abs/1911.03572).

## Requirements
0. GPU (Cuda 9.0+)
1. Python3 (<= 3.6.8)
2. Numpy
3. Sklearn
4. Pytorch (gpu/cpu) 1.4


### Download and install dependencies
To set up virtual environment and dependencies (on Linux):
```bash
cd DZip
python3 -m venv torch
source torch/bin/activate
bash install.sh
```

BSC Compressor is publicly available. To install on linux (optional/reduces the bootstrap model size)
```bash
git clone https://github.com/IlyaGrebnov/libbsc.git
cd libbsc && make
cp bsc ../coding-gpu/
```


##### ENCODING-DECODING
```bash 
cd coding-gpu
# Compress using the combined model (default usage of DZip)
bash compress.sh FILE.txt FILE.dzip com MODEL_PATH
# Compress using only the bootstrap model
bash compress.sh FILE.txt FILE.dzip bs MODEL_PATH
# Decompress using combined model (Only if compressed using combined mode)
bash decompress.sh FILE.dzip decom_FILE com MODEL_PATH
# Decompress using bootstrap model (Only if compressed using bs mode)
bash decompress.sh FILE.dzip decom_FILE bs MODEL_PATH
# Verify successful decompression
bash compare.sh FILE.txt decom_FILE
```

To compress/decompress bootstrap model with BSC
```bash
# Compress
./bsc e modelinput modeloutput -b128e2
# Decompress
./bsc d modeloutput modelinput
```

## Links to the Datasets with description and trained bootstrap models can be accesed [here](./Datasets.md)

### Please cite if you utilize the code in this repository.
```
@INPROCEEDINGS{9418692,
  author={Goyal, Mohit and Tatwawadi, Kedar and Chandak, Shubham and Ochoa, Idoia},
  booktitle={2021 Data Compression Conference (DCC)}, 
  title={DZip: improved general-purpose loss less compression based on novel neural network modeling}, 
  year={2021},
  volume={},
  number={},
  pages={153-162},
  doi={10.1109/DCC50243.2021.00023}}
```
