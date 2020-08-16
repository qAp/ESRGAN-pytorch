#!/bin/bash


python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --load=./parameters/psnr_based.pth --checkpoint_dir=./results/exp14  --batch_size=16 --epoch=0 --num_epoch=500 --is_perceptual_oriented=True --scale_factor=4 --distributed --fp16

#python src/train.py -n 1 -g 1 -nr 0 --input_dir=./datasets --resume=./results/exp11/last.pth --checkpoint_dir=./results/exp12  --batch_size=16 --epoch=0 --num_epoch=800 --fp16 --is_perceptual_oriented=False --scale_factor=4 


#python src/train.py -n 2 -g 1 -nr 0 --input_dir=./src/datasets --load=./results/checkpoints_test/checkpoint_4.pth --resume=./results/checkpoints_test/last.pth --checkpoint_dir=./results/checkpoints_test --is_perceptual_oriented=True --scale_factor=4 --batch_size=16 --epoch=0 --num_epoch=4 --fp16
#
