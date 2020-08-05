#!/bin/bash

mkdir -p results/checkpoints_6
cp results/checkpoints_5/*_19.pth results/checkpoints_6/.

python main.py --input_dir=/home/ubuntu/data --checkpoint_dir=results/checkpoints_6 --sample_dir=results/samples_6 --is_perceptual_oriented=True --scale_factor=4 --batch_size=16 --epoch=20 --num_epoch=60

python main.py --input_dir=/home/ubuntu/data --checkpoint_dir=results/checkpoints_6 --sample_dir=results/samples_6 --is_perceptual_oriented=False --scale_factor=4 --batch_size=16 --epoch=60 --num_epoch=100
