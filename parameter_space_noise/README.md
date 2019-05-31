# hallgato-vuchetichbalint

Paperspace P5000

```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev python-pip
pip3 install stable-baselines gym[Box2D] 
pip3 install gym[atari]

screen 
jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888
```

docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/root/code/stable-baselines,type=bind araffin/stable-baselines

AWS p3.2xlarge

```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev python-pip
pip install stable-baselines gym[Box2D]
pip install gym[atari]


screen 
jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888
```
AWS m5.24xlarge

```
sudo apt update && sudo apt install cmake libopenmpi-dev zlib1g-dev python-pip
pip install stable-baselines
pip install gym[Box2D]
pip install gym[atari]


screen 
jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888
```

docker exec -it test /bin/bash


## run experimental:
```
cd workspace
git clone https://github.com/hill-a/stable-baselines
# copy experimental setup ...

sudo docker pull araffin/stable-baselines
sudo docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/root/code/stable-baselines,type=bind araffin/stable-baselines
apt-get update && apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install gym[Box2D] gym[atari]
cd 
```


## check status:
```
import pandas as pd
df = pd.read_csv('monitor.csv', skiprows=1); df['l'].sum()
```


envs:
sparse:
https://gym.openai.com/envs/NChain-v0/



Please use one of the following commands to start the required environment with the framework of your choice:
for MXNet(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p36
for MXNet(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p27
for TensorFlow(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p36
for TensorFlow(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p27
for Theano(+Keras2) with Python3 (CUDA 9.0) _______________________________________________ source activate theano_p36
for Theano(+Keras2) with Python2 (CUDA 9.0) _______________________________________________ source activate theano_p27
for PyTorch with Python3 (CUDA 9.2 and Intel MKL) ________________________________________ source activate pytorch_p36
for PyTorch with Python2 (CUDA 9.2 and Intel MKL) ________________________________________ source activate pytorch_p27
for CNTK(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _________________________________ source activate cntk_p36
for CNTK(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _________________________________ source activate cntk_p27
for Caffe2 with Python2 (CUDA 9.0) ________________________________________________________ source activate caffe2_p27
for Caffe with Python2 (CUDA 8.0) __________________________________________________________ source activate caffe_p27
for Caffe with Python3 (CUDA 8.0) __________________________________________________________ source activate caffe_p35
for Chainer with Python2 (CUDA 9.0 and Intel iDeep) ______________________________________ source activate chainer_p27
for Chainer with Python3 (CUDA 9.0 and Intel iDeep) ______________________________________ source activate chainer_p36
for base Python2 (CUDA 9.0) __________________________________________________________________ source activate python2
for base Python3 (CUDA 9.0) __________________________________________________________________ source activate python3







ssh -i "balint_aws.pem" ubuntu@ec2-52-59-238-204.eu-central-1.compute.amazonaws.com


source activate tensorflow_p36

sudo apt update
sudo apt install cmake libopenmpi-dev zlib1g-dev python-pip htop
pip install stable-baselines
pip install gym[Box2D]
pip install gym[atari]

(reconnect)

ssh -i "balint_aws.pem" ubuntu@ec2-52-59-238-204.eu-central-1.compute.amazonaws.com

screen 
source activate tensorflow_p36
mkdir workspace
cd workspace

#<copies with filezilla>


check status:
awk 'FNR>2' dqn_e21/*/*/*/* | awk -F ',' '{ sum += $2 } END { print sum }'



awk 'FNR>2' ./*/*/monitor.csv | awk -F ',' '{ sum += $2 } END { print sum }'








