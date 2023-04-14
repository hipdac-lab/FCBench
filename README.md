# FCbench: Cross-Domain Benchmarking of Lossless Compression for Floating-point Data: Uniting HPC and Database Communities

We benchmarked eight CPU-based and five GPU-based lossless compression methods on 33 datasets from scientific simulation, time series, observation and database transactions domains.

## Test system
The experiments are carried out on a Chameleon Cloud compute node with 2 Intel(R) Xeon(R) Gold 6126 CPUs,
2.60GHz, 187 GB RAM. The compute node also has 1 Nvidia Quadro RTX 6000GPU with 24 GB GPU Memory.
The node compilers are GCC/G++9.4, CUDA 11.3, CMAKE 3.25.0 and python 3.8
- setup directories
```
mkdir code data experiments output software
```

## CPU-based methods
### fpzip
- compile
```
cd /home/cc/code/fpzip
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
- evaluate
```
cd /home/cc/code/fpzip
bash scripts/test_fpzip.sh
```

### pFPC
- compile
```
cd /home/cc/code/pFPC
gcc -O3 -pthread -std=c99 pFPC.c -o pfpc
```
- evaluate
```
cd /home/cc/code/pFPC
bash test_pfpc.sh
```

### SPDP
- compile
```
cd /home/cc/code/SPDP
gcc -O3 SPDP_11.c -o spdp
```
- evaluate
```
cd /home/cc/code/SPDP
bash test_spdp.sh
```

### Bitshuffle
- compile
```
cd /home/cc/code/bitshuffle
python -m venv ~/env4shuffle
source /home/cc/env4shuffle/bin/activate
pip install setuptools==62.3.3
pip install Cython
pip install numpy
python setup.py install --zstd
```
- evaluate
```
cd /home/cc/code/bitshuffle
bash scripts/test_bitshuffle.sh
```

### ndzip-CPU

### BUFF

### Gorilla

### Chimp

## GPU-based methods
### GFC

### MPC

### nvCOMP

### ndzip-GPU

### Dzip

## Datasets
### HPC
- msg-bt
- num-brain
- num-control
- rsim
- astro-mhd
- astro-pt
- miranda3d
- turbulance
- wave
- hurricane

### Time series
- citytemp
- ts-gas
- phone-gyro
- wesad-chest
- jane-street
- nyc-taxi
- gas-price
- solar-wind

### Observations
- acs-wht
- hdr-night
- hdr-palermo
- hst-wfc3-vuis
- hst-wfc3-ir
- spitzer-irac
- g24-78-usb
- jws-mirimage

### Database transactions
- tpcH-order
- tpcxBB-store
- tpcxBB-web
- tpcxH-lineitem
- tpcDS-catalog
- tpcDS-store
- tpcDS-web
