#!/bin/bash


python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --load=./parameters/psnr_based.pth --checkpoint_dir=./results/checkpoints_test --sample_dir=./results/samples_test --is_perceptual_oriented=False --scale_factor=4 --batch_size=16 --epoch=0 --num_epoch=2

python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --load=./results/checkpoints_test/checkpoint_4.pth --resume=./results/checkpoints_test/checkpoint_1.pth --checkpoint_dir=./results/checkpoints_test --sample_dir=./results/samples_test --is_perceptual_oriented=False --scale_factor=4 --batch_size=16 --epoch=0 --num_epoch=4
