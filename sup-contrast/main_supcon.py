from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np

#import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

from torch.utils.data import Dataset
from PIL import Image

"""try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass"""

from diffusers.utils import check_min_version, is_wandb_available
if is_wandb_available():
    import wandb


class MVIPDataset(Dataset):
    def __init__(
        self,
        transform=None,
        repeats=100,
        split="train",
    ):
        self.data_root = '/mnt/HDD/MVIP/sets'

        split_dir = {
            "train": "train_data",
            "val": "valid_data",
            "test": "test_data",
        }
        self.split = split_dir[split]

        self.image_paths = self.get_image_paths()
        self.num_images = len(self.image_paths)

        if set == "train":
            self._length = self.num_images * repeats
        else:
            self._length = self.num_images

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(512, scale=(0.2, 1.)),
                #transforms.Resize(512, Image.Resampling.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        # Label
        example["labels"] = torch.tensor([int(self.image_paths[i % self.num_images].split("/")[-6])])

        # Image
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["pixel_values"] = self.transform(image)

        return example
    
    def get_image_paths(self): # /mnt/HDD/MVIP/sets/class_name/train_data/0/0/cam0/0_rgb.png
        paths = []

        for class_name in [f for f in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, f))]:
            root = os.path.join(self.data_root, class_name, self.split)

            for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                for orientation in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                    for cam in [f for f in os.listdir(os.path.join(root, set, orientation)) if os.path.isdir(os.path.join(root, set, orientation, f))]:
                        for file in os.listdir(os.path.join(root, set, orientation, cam)):
                            if file.endswith("rgb.png"):
                                paths.append(os.path.join(root, set, orientation, cam, file))
        
        return paths
    
    def get_mean_std(self):
        mean = torch.zeros(3)
        std = torch.zeros(3)

        for image in self.image_paths:
            image = Image.open(image)
            image = self.transform(image)
            mean += torch.mean(image, dim=(1, 2))
            std += torch.std(image, dim=(1, 2))

        mean /= len(self.image_paths)
        std /= len(self.image_paths)

        return mean, std


def parse_args():
    parser = argparse.ArgumentParser('Arguments for training')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mvip'])
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)

    # Model
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--size', type=int, default=512, help='Parameter for RandomResizedCrop.')

    parser.add_argument('--save_freq', type=int, default=50,)
    parser.add_argument('--print_freq', type=int, default=10,)

    # Optimization
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='When to decay lr, as string seperated by ","')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # Method
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'])

    # Temperature
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for loss function.')

    # Other settings
    parser.add_argument('--cosine', action='store_true', help='Using cosine annealing.')
    parser.add_argument('--syncBN', action='store_true', help='Using synchronized batch normalization.')
    parser.add_argument('--warm', action='store_true', help='Warm-up for large batch training.')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs.')

    args = parser.parse_args()

    # Set specific arguments for dataset=mvip
    if args.dataset == 'mvip':
        assert args.experiment_name is not None, "--experiment_name is required for dataset=mvip."

        args.save_dir = os.path.abspath(os.path.join(
            os.path.dirname( __file__ ), '..', '_experiments', args.experiment_name, 'SupCon/models'
        ))

        args.logging_dir = os.path.abspath(os.path.join(
            os.path.dirname( __file__ ), '..', '_experiments', args.experiment_name, 'SupCon/logs'
        ))
    else:
        args.save_dir = './save/SupCon/{}_models'.format(args.dataset)
        args.logging_dir = './save/SupCon/{}_tensorboard'.format(args.dataset)

    # Set learning rate decay epochs from string argument
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    # Set model name for output
    args.model_name = '{}_{}_{}_lr={}_decay={}_bs={}_temp={}_trial={}'.\
        format(args.method, args.dataset, args.model, args.learning_rate,
               args.weight_decay, args.batch_size, args.temp, args.trial)

    if args.cosine:
        args.model_name = '{}_cosine'.format(args.model_name)

    # Warm-up for large-batch training,
    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.model_name = '{}_warm'.format(args.model_name)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    args.logging_dir = os.path.join(args.logging_dir, args.model_name)
    if not os.path.isdir(args.logging_dir):
        os.makedirs(args.logging_dir)

    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


# Construct data loader
def set_loader(args):
    print("Preparing dataloader...")
    if args.dataset == 'mvip':
        train_dataset = MVIPDataset(split="train")

        mean, std = train_dataset.get_mean_std()
        print("Mean: {}, Std: {}".format(mean, std))

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_dataset.transform = train_transform
    else:
        mean_std = {
            'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        }

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[args.dataset][0], std=mean_std[args.dataset][1]),
        ])

        train_dataset = datasets.CIFAR10(root=args.data_dir,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    
    print("Dataloader ready.")

    return train_loader


def set_model(args):
    model = SupConResNet(name=args.model)
    criterion = SupConLoss(temperature=args.temp)

    """# Enable synchronized batch normalization
    if args.syncBN:
        model = apex.parallel.convert_syncbn_model(model)"""

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, args):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start_time = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - start_time)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # Warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # Compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if args.method == 'SupCon':
            loss = criterion(features, labels)
        elif args.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('Contrastive method not supported: {}'.
                             format(args.method))

        # Update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - start_time)

        # Print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    args = parse_args()

    # Build data loader
    train_loader = set_loader(args)

    # Build model and criterion
    model, criterion = set_model(args)

    # Build optimizer
    optimizer = set_optimizer(args, model)

    # Init W&B logging
    run = wandb.init(
        project='SupCon',
        config={
            "dataset" : args.dataset,
            "model_name": args.model_name,
            "batch_size" : args.batch_size,
            "learning_rate" : args.learning_rate,
            "epochs" : args.epochs,
            "weight_decay" : args.weight_decay,
            "temperature" : args.temp,
            "cosine" : args.cosine,
        },
    )

    # Training routine
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # Train for one epoch
        start_time = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, args)
        end_time = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, end_time - start_time))

        # W&B logger
        wandb.log({
            "loss": loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # Save the last model
    save_file = os.path.join(
        args.save_dir, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)


if __name__ == '__main__':
    main()
