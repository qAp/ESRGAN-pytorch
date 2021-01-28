# ESRGAN-pytorch

  > ESRGAN-pytorch + distributed training

  This fork of [ESRGAN-pytorch](https://github.com/wonbeomjang/ESRGAN-pytorch.git) makes use of `apex` to enable optional training across multiple nodes with multiple GPUs and mixed precision.

## How to install
   ```
   git clone https://github.com/qAp/ESRGAN-pytorch.git
   cd ESRGAN-pytorch
   git install --editable .
   ```

   See [here](https://github.com/qAp/omdena_engie/blob/master/omdena_engie/06_ESRGAN-pytorch_training_colab.ipynb) for an example of how to install `apex` and get this package to work in Colab.  Note that on Colab there're no multiple nodes nor GPUs though.

## Usage

### General usage
   See the original repo's README.md for instructions on general usage of ESRGAN-pytorch.

### Distributed training
   To run distributed training, suppose you have two nodes, A and B, each with one GPU, and that you have selected node A as the master node.  Then,on all nodes, in `train.main`, edit the following 2 lines:

   ```python
   os.environ['MASTER_ADDR'] = '172.31.29.213'
   os.environ['MASTER_PORT'] = '8889'
   ```
   where `MASTER_ADDR` is the IP address of node A, and `MASTER_PORT` is a port that you have selected for communicating with other nodes.

   To start the training, execute the `train.py` script.  On node A, the master node, in addition to the usual arguments for training, supply the following:
   ```
   python train.py -n 2 -g 1 -nr 0 --distributed
   ```
   This says that there are a total of 2 nodes, the current node, node A, has node rank 0, which indicates that it's the master node.  `-g 1` indicates that there is only 1 GPU on this node.

   Once you have executed `train.py` on node A, go onto node B and execute the same script with:
   ```
   python train.py -n 2 -g 1 -nr 1 --distributed
   ```
   which indicates that node B has node rank of 1 and that it also has only 1 GPU.  The total number of nodes stays at 2.

### Mixed precision training
    To use mixed precision in training, supply the `--fp16` argument to `train.py`.

