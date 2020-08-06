#!/bin/bash


python main.py --input_dir=datasets --checkpoint_dir=results/checkpoints_test --sample_dir=results/samples_test --is_perceptual_oriented=True --scale_factor=4 --batch_size=16 --epoch=0 --num_epoch=1

python main.py --input_dir=datasets --checkpoint_dir=results/checkpoints_test --sample_dir=results/samples_test --is_perceptual_oriented=False --scale_factor=4 --batch_size=16 --epoch=1 --num_epoch=2
