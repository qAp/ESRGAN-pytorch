#!/bin/bash


python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --resume=./results/exp9/best.pth --checkpoint_dir=./results/exp10  --batch_size=16 --epoch=0 --num_epoch=1000 --fp16 --is_perceptual_oriented=False --scale_factor=4 --g_content_loss_factor=1e-2

#python src/train.py -n 2 -g 1 -nr 0 --input_dir=./src/datasets --load=./results/checkpoints_test/checkpoint_4.pth --resume=./results/checkpoints_test/last.pth --checkpoint_dir=./results/checkpoints_test --is_perceptual_oriented=True --scale_factor=4 --batch_size=16 --epoch=0 --num_epoch=4 --fp16
#
