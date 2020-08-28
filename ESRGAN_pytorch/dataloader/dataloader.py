from torch.utils.data import DataLoader
from ESRGAN_pytorch.dataloader.datasets import Datasets
import torch


def get_loader(args):
    train_dataset = Datasets(args.image_size, args.scale_factor, args.input_dir)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, shuffle=True)
    return train_loader
