#!/bin/bash




# Example: Start PSNR-based training.
#python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --load=./parameters/gan_based.pth --checkpoint_dir=./results/exp16  --batch_size=16 --epoch=0 --num_epoch=350 --is_perceptual_oriented=True --scale_factor=4 --fp16 --distributed

# Example: Resume  PSNR-based training.
#python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --resume=./results/exp14/last.pth --checkpoint_dir=./results/exp14  --batch_size=16 --epoch=0 --num_epoch=350 --is_perceptual_oriented=True --scale_factor=4 --fp16 --distributed

# Example: Start GAN-based training.
python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --load=./parameters/gan_based.pth --checkpoint_dir=./results/exp17  --batch_size=16 --epoch=0 --num_epoch=500 --is_perceptual_oriented=False --scale_factor=4 --fp16 --distributed

# Example: Resume GAN-based training.
# python src/train.py -n 2 -g 1 -nr 0 --input_dir=./datasets --resume=./results/exp15/last.pth --checkpoint_dir=./results/exp15  --batch_size=16 --epoch=0 --num_epoch=200 --is_perceptual_oriented=False --scale_factor=4 --fp16 --distributed

#python src/train.py -n 1 -g 1 -nr 0 --input_dir=./datasets --resume=./results/exp11/last.pth --checkpoint_dir=./results/exp12  --batch_size=16 --epoch=0 --num_epoch=800 --fp16 --is_perceptual_oriented=False --scale_factor=4 


#python src/train.py -n 2 -g 1 -nr 0 --input_dir=./src/datasets --load=./results/checkpoints_test/checkpoint_4.pth --resume=./results/checkpoints_test/last.pth --checkpoint_dir=./results/checkpoints_test --is_perceptual_oriented=True --scale_factor=4 --batch_size=16 --epoch=0 --num_epoch=4 --fp16
#
