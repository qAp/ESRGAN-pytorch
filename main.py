from src.train import Trainer
from dataloader.dataloader import get_loader
import os
from config.config import get_config
import pprint




def main():
    args = get_config()
    # make directory not existed
    if args.checkpoint_dir is None:
        args.checkpoint_dir = 'checkpoints'
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    print(f"ESRGAN start")

    data_loader = get_loader(args.image_size, args.scale_factor,
                             args.batch_size, args.sample_batch_size,
                             args.input_dir)
    trainer = Trainer(args, data_loader)
    trainer.train()


if __name__ == "__main__":

    pprint.pprint(args)
    main(args)
