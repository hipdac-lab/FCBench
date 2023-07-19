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

## Evaluate and Compare
### Run all compression algorithms and output results
```
export MYDATA=path/to/data
export MYSCRATCH=path/to/code
cd scripts
bash eval_all.sh
```
### Prepare output results to intermediate results
```
bash awk_all.sh
```
### read intermediate results and display results
```
python fcb_res.py
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
gcc -O3 -pthread -std=c99 pFPC_block.c -o pfpcb
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
gcc -O3 SPDP_block.c -o spdpb
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
- compile
```
cd /home/cc/code/ndzip
whichgcc=$(which gcc)
whichgpp=$(which g++)
boostdir=/home/cc/software/boost
whichnvcc=$(which nvcc)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=75 \
	-DCMAKE_BUILD_TYPE=Release \
	-DZIP_WITH_MT=YES \
	-DNDZIP_WITH_CUDA=YES \
	-DCMAKE_CXX_FLAGS="-march=native" \
	-DCMAKE_C_COMPILER=$whichgcc \
	-DCMAKE_CXX_COMPILER=$whichgpp \
	-DCMAKE_CUDA_COMPILER=$whichnvcc \
	-DBoost_INCLUDE_DIR=$boostdir/include \
	-DBoost_LIBRARY_DIRS=$boostdir/lib
cmake --build build -j
```
- evaluate
```
cd /home/cc/code/ndzip
bash scripts/batch_ndzip_cpu.sh
```

### BUFF
- compile
```
cd /home/cc/code/buff/database
rustup default nightly
cargo +nightly build --release  --package buff --bin comp_profiler
```
- evaluate
```
cd /home/cc/code/buff/
bash test_buff_p10.sh
```

### Gorilla
- compile
```
cd /home/cc/code/influxdb
rustup default 1.53.0
make
```
- evaluate
```
cd /home/cc/code/influxdb
git checkout gorilla
go clean -testcache
go test  -test.timeout 0 -run TestCompress_XC2 -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1
```

### Chimp
- Already compiled because both Gorilla and Chimp are parts of influxdb
- evaluate
```
cd /home/cc/code/influxdb
git checkout chimp128
go clean -testcache
go test  -test.timeout 0 -run TestCompress_XC2 -v github.com/influxdata/influxdb/v2/tsdb/engine/tsm1
```

## GPU-based methods
### GFC
- compile
```
cd /home/cc/code/GFC
nvcc -O3 -arch=sm_60 GFC_22.cu -o GFC
```
- evaluate
```
cd /home/cc/code/GFC
bash test_GFC.sh
```

### MPC
- compile
```
cd /home/cc/code/MPC
nvcc -O3 -arch=sm_60 MPC_float_12.cu -o MPC_float
nvcc -O3 -arch=sm_60 MPC_double_12.cu -o MPC_double
```
- evaluate
```
cd /home/cc/code/MPC
bash test_mpc.sh
```

### nvCOMP
- does not need to compile
- evaluate
```
cd /home/cc/code/nvbench
bash batch_nvcomp.sh
```

### ndzip-GPU
- Already compiled in ndzip-CPU
- evaluate
```
cd /home/cc/code/ndzip
bash scripts/batch_ndzip_gpu.sh
```

### Dzip
- compile
```
python -m venv ~/env4dzip
source ~/env4dzip/bin/activate
cd /home/cc/code/Dzip-torch
bash install.sh
```

## Datasets
### Down the data
We recommend ```gdown``` to download a large file from Google Drive.
Use ```pip install gdown``` to install.
#### Download the folders
- HPC, TS and OBS datasets
```
gdown https://drive.google.com/drive/folders/1jdnzwvT1hya8XYdEJ7QuqUw3ALbQozc7 --folder
```
- DB (TPC synthetic data)
```
gdown https://drive.google.com/drive/folders/1WKvzMErKfhqAGRkJhqXZScH15kPHUxnG --folder
```
#### Download individual files

```
file-id=example_1i0AK1sLjYBnISDU9e8_FbV0OsowExHV3
gdown https://drive.google.com/uc?id=$file-id
```

### HPC
| file-name   | file-id |
| ----------- | ----------- |
| msg-bt|	 15S7iTr_Yoo6oVv5TOemah0wP1K7VX5R1 |
| num-brain|	1D2WEJonO3GWQwAQxSokO6Pn4kffalhCy |
| num-control|	13Lpx_S0W4K5hBMOvyW61PFUOv9BXGOMN |
| rsim|	1C6opL2ZyJyU4074uc9T9eJBnyTvhW16- |
| astro-mhd|1gp2pUEtr8FP3g7hbu4EhDYtTyg2eVoBr|
| astro-pt|1ZI6h-8OOW2h7DG9L4P9tIGrUo_KBMH2R|
| miranda3d|	1jTCH1i_1w_zGvfBydT1Ac-kJK-H4DZHS |
| turbulance|	11MNFi9pGpU9IDw1QSZrA-y2oY5xPNyzs |
| wave|	1jokNzOoEHOrXpCw8glhGzA_1lJxedKx2 |
| hurricane|	1h48gO2JNCNWMVaEPsIHPBkNllgtHDzcf	|

### Time series
| file-name   | file-id |
| ----------- | ----------- |
| citytemp: 	 |1S6MN7A0BxbQJ0MWZWF7HPdFn2-O6fLlb	|
| ts-gas |	1P2rLx16bVLJZEdnwGYipXk0aHlkOJtQl |
| phone-gyro |	18WPrgYKUKg1vOuKatDgAQu7_2iDQ6EnK |
| wesad-chest |	1v1Mz4ka_kmwFF5Bcb7QXg8pAYZ43_Ptv |
| jane-street |	19JQgBJaLeHBaCV6G-Tcpqye8BQVWlqFt |
| nyc-taxi |	1ODXvl_gsohxv4z29aNfL0gk458fiZYYO |
| gas-price |	1n4ihLBaIbQji2iMjlAzTDL1_ryhG6E5z |
| solar-wind |	1sVAEV0wLdfrrXFm6uQKvYU0412vKlaQa |

### Observations
| file-name   | file-id |
| ----------- | ----------- |
| acs-wht 			 |	1i0AK1sLjYBnISDU9e8_FbV0OsowExHV3   |
| hdr-night | 1zgHWgF3xYTeQXHY04P6vewcgzMgsQQrZ |
| hdr-palermo | 1d624EAKKy9KoZ1g2exRy9v6BaHM18GKH |
| hst-wfc3-vuis  | 11CSs6GMg6H_IIXGd_Rzrg6AjbC4fNLrB  |
| hst-wfc3-ir | 1IqngOcGb-Kd8_3qevjwIp3oR88xHTJLM |
| spitzer-irac |  167QBkioAKm0lDUn9PaVsCG-5mPxpawSI |
| g24-78-usb	 	 |	12zF9g7oWo9kkIB0Y0QtOWBW0dZt2J_2Q  |
| jws-mirimage |  1SIB0wg5SmH0L3i8neoIwLHazuykwcNYK |

### Database transactions
| file-name   | file-id |
| ----------- | ----------- |
| tpcH-order  |  1BFbsUJjt9nS1n9wIFAEng07Mpgyozjgz |
| tpcxBB-store  |  1QIAtpHc-WLVHa5KFj2UgzcGS9tHIv6L1 |
| tpcxBB-web  |  1TVdj2kirdpIShuFMnfKUWBowMyeiC3_w |
| tpcxH-lineitem  |  1RNU6Xg7tSei34bS8fAVRTRU_jpVYcmGK |
| tpcDS-catalog  |  1W_wDxENpUOZ-uIyAmZfhu-2Pr4Z1VP_s |
| tpcDS-store  |  1L-ED3uP1oXFzoVLYJ17EvEq7ObqYhT7k |
| tpcDS-web  |  1_4RgPBsOr57wkiM9wkVXBptVTRd4CYl7 |

## Experiment results
### Compression ratios
<img src="https://user-images.githubusercontent.com/130711868/232258352-fbf11c8d-cb80-4a7f-a7c4-a3881eaa0600.png" width="50%">
<img src="https://user-images.githubusercontent.com/130711868/232258373-fc4b408d-2dda-4895-b49d-89cf6c80538f.png" width="49%">

### Compression throughputs
<img src="https://user-images.githubusercontent.com/130711868/232258394-d4e0de8c-894d-4bea-a24f-4ef4f1cc3b57.png" width="50%">
<img src="https://user-images.githubusercontent.com/130711868/232258398-25c13e56-6f33-401b-9f71-b43f05a02c00.png" width="49%">

### Decompression throughputs
<img src="https://user-images.githubusercontent.com/130711868/232258403-34a78480-1965-4d16-88ea-4c177377a83e.png" width="50%">
<img src="https://user-images.githubusercontent.com/130711868/232258407-fe89022b-641e-4605-913f-6cc113aea4e0.png" width="49%">

### Difference of throughputs
<img src="https://user-images.githubusercontent.com/130711868/232258423-317a31ec-eaba-405b-bc8c-f85984306899.png" width="50%">

### Roofline model of CPU-based methods
<img src="https://user-images.githubusercontent.com/130711868/232258440-62220371-4f97-417a-9b8e-acec994bfd34.png" width="50%">

### Roofline model of GPU-based methods
<img src="https://user-images.githubusercontent.com/130711868/232258448-3ca8de9b-266c-416a-8ecb-c9871f3ff000.png" width="50%">
