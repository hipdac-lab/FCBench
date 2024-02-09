# Install docker
## Step 1: Clean previous installed docker
```
sudo docker --version
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```
## Step 2: Download, install and check
```
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```

# Use the pre-built fcbench image
## Step 1: Pull the docker image
Assuming [docker](https://docs.docker.com/get-docker/) has been installed, please run the following command to pull our prepared docker image (https://hub.docker.com/r/xinyu125/fcbench) from DockerHub:
```
docker pull xinyu125/fcbench
```
The image comes with the compiled executables for this benchmark study. 

## Step 2: Evaluation

### Setup data folder and download the datasets into it.
```
mkdir data
```

### Launch the docker image:
```
sudo docker run --rm --gpus all -v $(pwd)/data:/opt/data -it xinyu125/fcbench:0.1 /bin/bash
```

### Evaluate the methods:
```
cd /opt/scripts
bash eval_all.sh
```
The whole evaluation will take about 3~4 hours to run all the methods. You can also evaluate each method individually. The executables are already compiled. Thus you can directly use the test_XXX.sh in each corresponding directory. More detailed description of using such scripts will be listed in the Method 2 (Build from source). 

### Extract the results:
```
cd /opt/scripts
bash awk_all.sh
```

### Display the results. The "work" parameter are "cr", "ct" and "dt" for compression ratio, conpression throughput and decompression throughput respectively.
```
cd /opt/scripts
python3 fcb_res.py --work=cr
```
BUFF results are not displayed because of bit-packing errors on some datasets. We are investigating the issue and will fix soon.

# Build the image from scratch
## build from source
```
cd docker
sudo docker build -t your-dockerhub-username/fcbench:0.1 -f chameleon.Dockerfile.1 src
```
