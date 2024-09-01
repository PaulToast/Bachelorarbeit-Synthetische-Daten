from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

from datasets import MVIPDataset

from diffusers.utils import check_min_version, is_wandb_available
if is_wandb_available():
    import wandb


def parse_args():
    parser = argparse.ArgumentParser('Arguments for training')

    parser.add_argument('--experiment_name', type=str, default=None, help='Output directory name for experiment.')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs.')

    # Data
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mvip'])
    parser.add_argument('--data_dir', type=str, default=None)

    parser.add_argument('--aug_method', type=str, default=None, choices=[None, 'positive', 'both'])
    parser.add_argument('--aug_experiment', type=str, default="mvip-v9-final")
    parser.add_argument('--aug_name_positive', type=str, default="aug=0.2_ex=16_num=4_g=15")
    parser.add_argument('--aug_name_negative', type=str, default="aug=0.5_ex=16_num=4_g=15")
    #parser.add_argument('--aug_ex_positive', type=int, default=-1)
    #parser.add_argument('--aug_ex_negative', type=int, default=8)

    # Training
    parser.add_argument('--model', type=str, default='resnet50')

    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'])

    parser.add_argument('--size', type=int, default=224, help='Size for RandomResizedCrop.') #32

    parser.add_argument('--epochs', type=int, default=110) #1000
    parser.add_argument('--batch_size', type=int, default=16) #256
    parser.add_argument('--num_workers', type=int, default=4) #16

    parser.add_argument('--lr', type=float, default=0.002) #0.05
    parser.add_argument('--lr_warmup', action='store_true', help='Learning rate warm-up for large batch training.')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='When to decay lr, as string separated by comma')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_cosine', action='store_true', help='Using cosine annealing.')

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for loss function.')

    #parser.add_argument('--syncBN', action='store_true', help='Using synchronized batch normalization.')

    # Output & logging
    parser.add_argument('--save_freq', type=int, default=10, help='Save model every N epochs.') #50
    parser.add_argument('--print_freq', type=int, default=10, help='Print training progress every N steps.')

    args = parser.parse_args()

    args.model_name = '{}_{}_{}_trial={}_lr={}_decay={}_bs={}_temp={}'.\
        format(args.method, args.dataset, args.model, args.trial, args.lr,
               args.weight_decay, args.batch_size, args.temp)

    # Set aug directories
    if args.aug_method is not None:
        assert args.aug_experiment is not None

        if args.aug_method == 'positive':
            assert args.aug_name_positive is not None
            args.aug_dir_positive = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da-fusion/output/{args.aug_experiment}/{args.aug_name_positive}'
            ))
            args.aug_dir_negative = None
        elif args.aug_method == 'both':
            assert args.aug_name_positive is not None \
                and args.aug_name_negative is not None
            args.aug_dir_positive = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da-fusion/output/{args.aug_experiment}/{args.aug_name_positive}'
            ))
            args.aug_dir_negative = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da-fusion/output/{args.aug_experiment}/{args.aug_name_negative}'
            ))
    else:
        args.aug_dir_positive = None
        args.aug_dir_negative = None

    # Set-up learning rate
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if args.lr_cosine:
        args.model_name = '{}_cosine'.format(args.model_name)

    if args.batch_size > 256:
        args.lr_warmup = True
    if args.lr_warmup:
        args.model_name = '{}_warmup'.format(args.model_name)
        args.lr_warmup_from = 0.0001
        args.lr_warmup_epochs = 1
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
        else:
            args.lr_warmup_to = args.lr
    
    # Set-up output directory
    args.save_dir = os.path.abspath(f'output/{args.experiment_name}/{args.model_name}')
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def set_loader(args, split="train"):
    # Set-up transforms
    mean_std = {
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        'mvip': ([0.4213, 0.4252, 0.4242], [0.1955, 0.1923, 0.1912]),
    }

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.6, 1.)), # ratio=(1.0, 1.0))
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15.0),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[args.dataset][0], std=mean_std[args.dataset][1]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[args.dataset][0], std=mean_std[args.dataset][1]),
        ])

    # Set-up dataset
    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root=args.data_dir,
                                   train=(split == "train"),
                                   transform=TwoCropTransform(transform),
                                   download=True)
    elif args.dataset == "cifar100":
        dataset = datasets.CIFAR100(root=args.data_dir,
                                    train=(split == "train"),
                                    transform=TwoCropTransform(transform),
                                    download=True)
    elif args.dataset == "mvip":
        if split == "ood":
            aug_mode = "negative_only"
        elif split == "train":
            aug_mode = args.aug_method
        else:
            aug_mode = None
        
        dataset = MVIPDataset(split=split,
                              aug_mode=aug_mode,
                              aug_dir_positive=args.aug_dir_positive,
                              aug_dir_negative=args.aug_dir_negative,
                              #aug_ex_positive=args.aug_ex_positive,
                              #aug_ex_negative=args.aug_ex_negative,
                              size=args.size,
                              transform=TwoCropTransform(transform))

    # Build dataloader
    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=sampler)

    return dataloader


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

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

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

        # Update metric & log
        losses.update(loss.item(), bsz)
        
        wandb.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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

def validate(val_loader, model, criterion, args):
    """One epoch validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # Compute loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if args.method == 'SupCon':
                loss = criterion(features, labels)
            elif args.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('Contrastive method not supported: {}'.format(args.method))

            # Update metric & log
            losses.update(loss.item(), bsz)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (idx + 1) % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses))
                sys.stdout.flush()

    return losses.avg


def main():
    args = parse_args()

    # Build dataloaders
    print("Preparing dataloaders...")

    train_loader = set_loader(args, "train")
    val_loader = set_loader(args, "val")

    print(f"Dataloaders ready. Train length: {train_loader.__len__()}, Validation length: {val_loader.__len__()}")

    # Build model and criterion
    model, criterion = set_model(args)

    # Build optimizer
    optimizer = set_optimizer(args, model)

    # Init W&B logging
    run = wandb.init(
        project=args.experiment_name,
        config={
            "dataset" : args.dataset,
            "model_name": args.model_name,
            "image_size" : args.size,
            "batch_size" : args.batch_size,
            "epochs" : args.epochs,
            "lr" : args.lr,
            "lr_warmup" : args.lr_warmup,
            "lr_cosine" : args.lr_cosine,
            "weight_decay" : args.weight_decay,
            "temperature" : args.temp,
        },
    )

    # Training routine
    print("Initiating training...")

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # Train & validate for one epoch
        start_time = time.time()

        avg_train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        avg_val_loss = validate(val_loader, model, criterion, args)

        end_time = time.time()

        print('epoch {}, total time {:.2f}, average train loss: {}, average validation loss: {}'.format(
            epoch, end_time - start_time, avg_train_loss, avg_val_loss))

        # Log average epoch loss
        wandb.log({
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
        })

        # Save model
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # Save the last model
    save_file = os.path.join(args.save_dir, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)


if __name__ == '__main__':
    main()
