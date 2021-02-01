# ESRGAN-pytorch

  > ESRGAN-pytorch + Nvidia/apex

  This fork of [wonbeomjang/ESRGAN-pytorch](https://github.com/wonbeomjang/ESRGAN-pytorch.git) incorporates [Nvidia/apex](https://github.com/NVIDIA/apex)(`apex`) to enable optional training across multiple nodes with multiple GPUs and/or with mixed precision.

## How to install
   ```
   git clone https://github.com/qAp/ESRGAN-pytorch.git
   cd ESRGAN-pytorch
   git install --editable .
   ```

   See [here](https://github.com/qAp/omdena_engie/blob/master/omdena_engie/06_ESRGAN-pytorch_training_colab.ipynb) for an example of how to install `apex` and get this package to work in Colab, though note that on Colab there're no multiple nodes nor GPUs.

## Usage

### General usage
   See the original repo's [README.md](https://github.com/wonbeomjang/ESRGAN-pytorch/blob/master/README.md) for instructions on general usage of ESRGAN-pytorch.

### Distributed training
   To run distributed training, suppose you have two nodes, A and B, each with one GPU.
   
   1. Choose one of the nodes to be the master node. (Suppose you have chosen node A.)
   2. Make sure all the data and packages required for training, including this one, have been installed on all nodes.
   3. On all nodes, in `train.main` of this package, edit the following 2 lines:
   ```python
   os.environ['MASTER_ADDR'] = '172.31.29.213'
   os.environ['MASTER_PORT'] = '8889'
   ```
   where `MASTER_ADDR` is the IP address of node A, and `MASTER_PORT` is a port that you have selected for communicating with other nodes.  
   4. To start the training, execute the `train.py` script on the master node (node A).  In addition to the usual arguments for training, supply the following:
   ```
   python train.py -n 2 -g 1 -nr 0 --distributed
   ```
   This says that there are a total of 2 nodes, the current node, node A, has node rank 0, which indicates that it's the master node.  `-g 1` indicates that there is only 1 GPU on this node.  
   5. Execute `train.py` on other nodes.  e.g. Here, go onto node B and execute the same script with:
   ```
   python train.py -n 2 -g 1 -nr 1 --distributed
   ```
   which indicates that node B has node rank of 1 and that it also has only 1 GPU.  The total number of nodes stays at 2.

### Mixed precision training
   To use mixed precision in training, supply the `--fp16` argument to `train.py`.

## Helpful links
- [*Amazon AWS Setup* by Nathan Inkawhich](https://tutorials.pytorch.kr/beginner/aws_distributed_training_tutorial.html#amazon-aws-setup)
- [*Distributed data parallel training in Pytorch* by yangkky](https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)
