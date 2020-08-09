import os
from glob import glob
from pathlib import Path
import torch, torch.nn as nn
from torch.optim.adam import Adam
from torchvision.utils import save_image
import torch.multiprocessing as mp, torch.distributed as dist
import apex
from loss.loss import PerceptualLoss
from config.config import parse_args
from model.ESRGAN import ESRGAN
from model.Discriminator import Discriminator
from dataloader.dataloader import get_loader


class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.data_loader = data_loader

        if args.is_perceptual_oriented:
            self.lr = args.p_lr
            self.content_loss_factor = args.p_content_loss_factor
            self.perceptual_loss_factor = args.p_perceptual_loss_factor
            self.adversarial_loss_factor = args.p_adversarial_loss_factor
            self.decay_batch_size = args.p_decay_batch_size
        else:
            self.lr = args.g_lr
            self.content_loss_factor = args.g_content_loss_factor
            self.perceptual_loss_factor = args.g_perceptual_loss_factor
            self.adversarial_loss_factor = args.g_adversarial_loss_factor
            self.decay_batch_size = args.g_decay_batch_size

        self.build_model(args)
        self.build_optimizer(args)
        self.initialize_model_opt_fp16()
        self.parallelize_model()
        self.history = {n:[] for n in ['adversarial_loss', 'discriminator_loss',
                                       'perceptual_loss', 'content_loss',
                                       'generator_loss']}
        if args.load: self.load_model(args)
        if args.resume: self.resume(args)
        self.build_scheduler(args)

    def train(self, args):
        total_step = len(self.data_loader)
        adversarial_criterion = nn.BCEWithLogitsLoss().cuda()
        content_criterion = nn.L1Loss().cuda()
        perception_criterion = PerceptualLoss().cuda()
        self.generator.train()
        self.discriminator.train()

        for epoch in range(args.epoch, args.num_epoch):
            sample_dir_epoch = Path(args.checkpoint_dir)/'sample_dir'/str(epoch)
            sample_dir_epoch.mkdir(exist_ok=True, parents=True)

            for step, image in enumerate(self.data_loader):
                low_resolution = image['lr'].cuda()
                high_resolution = image['hr'].cuda()

                real_labels = torch.ones((high_resolution.size(0), 1)).cuda()
                fake_labels = torch.zeros((high_resolution.size(0), 1)).cuda()

                ##########################
                #   training generator   #
                ##########################
                self.optimizer_generator.zero_grad()
                fake_high_resolution = self.generator(low_resolution)

                score_real = self.discriminator(high_resolution)
                score_fake = self.discriminator(fake_high_resolution)
                discriminator_rf = score_real - score_fake.mean()
                discriminator_fr = score_fake - score_real.mean()

                adversarial_loss_rf = adversarial_criterion(
                    discriminator_rf, fake_labels)
                adversarial_loss_fr = adversarial_criterion(
                    discriminator_fr, real_labels)
                adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                perceptual_loss = perception_criterion(
                    high_resolution, fake_high_resolution)
                content_loss = content_criterion(
                    fake_high_resolution, high_resolution)

                generator_loss = (adversarial_loss * self.adversarial_loss_factor + 
                                  perceptual_loss * self.perceptual_loss_factor + 
                                  content_loss * self.content_loss_factor)

                with apex.amp.scale_loss(
                        generator_loss, self.optimizer_generator) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer_generator.step()

                ##########################
                # training discriminator #
                ##########################

                self.optimizer_discriminator.zero_grad()

                score_real = self.discriminator(high_resolution)
                score_fake = self.discriminator(fake_high_resolution.detach())
                discriminator_rf = score_real - score_fake.mean()
                discriminator_fr = score_fake - score_real.mean()

                adversarial_loss_rf = adversarial_criterion(
                    discriminator_rf, real_labels)
                adversarial_loss_fr = adversarial_criterion(
                    discriminator_fr, fake_labels)
                discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                with apex.amp.scale_loss(discriminator_loss,
                                         self.optimizer_discriminator) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer_discriminator.step()

                self.lr_scheduler_generator.step()
                self.lr_scheduler_discriminator.step()
                
                if step % 1000 == 0:
                    print(f"[Epoch {epoch}/{args.num_epoch}] "
                          f"[Batch {step}/{total_step}] "
                          f"[D {discriminator_loss.item():.4f}] "
                          f"[G {generator_loss.item():.4f}] "
                          f"[adversarial "
                          f"{(adversarial_loss.item()*self.adversarial_loss_factor:.4f)}]"
                          f"[perceptual "
                          f"{perceptual_loss.item()*self.perceptual_loss_factor:.4f}]"
                          f"[content "
                          f"{content_loss.item()*self.content_loss_factor:.4f}]"
                          f"")
                    if step % 5000 == 0:
                        result = torch.cat((high_resolution, fake_high_resolution), 2)
                        save_image(result, sample_dir_epoch/f"SR_{step}.png")

            self.history['adversarial_loss'].append(
                adversarial_loss.item()*self.adversarial_loss_factor)
            self.history['discriminator_loss'].append(discriminator_loss.item())
            self.history['perceptual_loss'].append(
                perceptual_loss.item()*self.perceptual_loss_factor)
            self.history['content_loss'].append(
                content_loss.item()*self.content_loss_factor)
            self.history['generator_loss'].append(generator_loss.item())
                        
            torch.save(
                {'epoch':epoch, 'history':self.history,
                 'g_state_dict':self.generator.state_dict(),
                 'd_state_dict':self.discriminator.state_dict(),
                 'opt_g_state_dict':self.optimizer_generator.state_dict(),
                 'opt_d_state_dict':self.optimizer_discriminator.state_dict(),
                 'amp':apex.amp.state_dict(),
                 'args':self.args},
                os.path.join(args.checkpoint_dir, f'last.pth'))

    def build_model(self, args):
        self.generator = ESRGAN(3, 3, 64, scale_factor=args.scale_factor).cuda()
        self.discriminator = Discriminator().cuda()

    def build_optimizer(self, args):
        self.optimizer_generator = Adam(
            self.generator.parameters(), lr=self.lr,
            betas=(args.b1, args.b2), weight_decay=args.weight_decay)
        self.optimizer_discriminator = Adam(
            self.discriminator.parameters(), lr=self.lr,
            betas=(args.b1, args.b2), weight_decay=args.weight_decay)

    def initialize_model_opt_fp16(self):
        self.generator, self.optimizer_generator = apex.amp.initialize(
            self.generator, self.optimizer_generator, opt_level='O2')
        self.discriminator, self.optimizer_discriminator = apex.amp.initialize(
            self.discriminator, self.optimizer_discriminator, opt_level='O2')

    def parallelize_model(self):
        self.generator = apex.parallel.DistributedDataParallel(
            self.generator, delay_allreduce=True)
        self.discriminator = apex.parallel.DistributedDataParallel(
            self.discriminator, delay_allreduce=True)        

    def build_scheduler(self, args):
        print('Building scheduler', args.epoch)
        self.lr_scheduler_generator = torch.optim.lr_scheduler.StepLR(
            self.optimizer_generator, self.decay_batch_size,
            last_epoch=args.epoch if args.resume else -1)
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
            self.optimizer_discriminator, self.decay_batch_size,
            last_epoch=args.epoch if args.resume else -1)
        
    def load_model(self, args):
        path_to_load = Path(args.load)
        if path_to_load.is_file():
            checkpoint = torch.load(
                path_to_load, map_location=lambda storage, loc: storage.cuda())
            if checkpoint['g_state_dict'] is not None:
                self.generator.load_state_dict(checkpoint['g_state_dict'])
                print(f'[*] Loading generator parameters from {path_to_load}')
            if checkpoint['d_state_dict'] is not None:
                self.discriminator.load_state_dict(checkpoint['d_state_dict'])
                print(f'[*] Loading discriminator parameters from {path_to_load}')
            if checkpoint['amp'] is not None:
                apex.amp.load_state_dict(checkpoint['amp'])
        else:
            print(f'[!] No checkpoint found at {path_to_load}')
            
    def resume(self, args):
        path_to_resume = Path(args.resume)
        if path_to_resume.is_file():
            cpt = torch.load(
                path_to_resume, map_location=lambda storage, loc: storage.cuda())
            if cpt['epoch'] is not None: args.epoch = cpt['epoch'] + 1
            if cpt['history'] is not None: self.history = cpt['history']
            if cpt['g_state_dict'] is not None:
                self.generator.load_state_dict(cpt['g_state_dict'])
                print(f'[*] Loading generator parameters from {path_to_resume}')
            if cpt['d_state_dict'] is not None:
                self.discriminator.load_state_dict(cpt['d_state_dict'])
                print(f'[*] Loading discriminator parameters from {path_to_resume}')
            if cpt['opt_g_state_dict'] is not None:
                self.optimizer_generator.load_state_dict(cpt['opt_g_state_dict'])
                print(f'[*] Loading generator optmizer parameters from {path_to_resume}')
            if cpt['opt_d_state_dict'] is not None:
                self.optimizer_discriminator.load_state_dict(cpt['opt_d_state_dict'])
                print(f'[*] Loading discriminator optmizer parameters from {path_to_resume}')
            if cpt['amp'] is not None:
                apex.amp.load_state_dict(cpt['amp'])
        else:
            raise ValueError(f'[!] No checkpoint to resume from at {path_to_resume}')
            
                
def train(gpu, args):
    print('Start of train():', gpu)
    args.rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=args.rank)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    
    if args.checkpoint_dir is None:
        args.checkpoint_dir = 'checkpoints'
    if not os.path.exists(args.checkpoint_dir): 
        os.makedirs(args.checkpoint_dir)

    print(f"ESRGAN start")
    print(args)

    data_loader = get_loader(args)
    trainer = Trainer(args, data_loader)
    trainer.train(args)
            

def main():
    args = parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '172.31.29.213'
    os.environ['MASTER_PORT'] = '8889'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__=='__main__':
    main()
