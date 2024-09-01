from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from main_linear import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet

from datasets import MVIPDataset

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from diffusers.utils import check_min_version, is_wandb_available
if is_wandb_available():
    import wandb

def parse_args():
    parser = argparse.ArgumentParser('Arguments for training')

    parser.add_argument('--experiment_name', type=str, default=None, help='Output directory name for experiment.')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs.')

    # Data
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data_dir', type=str, default=None)

    parser.add_argument('--aug_method', type=str, default=None, choices=[None, 'positive'])
    parser.add_argument('--aug_experiment', type=str, default="mvip-v9-final")
    parser.add_argument('--aug_name_positive', type=str, default="aug=0.2_ex=16_num=4_g=15")
    parser.add_argument('--aug_name_negative', type=str, default="aug=0.5_ex=16_num=4_g=15")

    # Training
    parser.add_argument('--model', type=str, default='resnet50')

    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')

    parser.add_argument('--lr', type=float, default=0.2) #0.2
    parser.add_argument('--lr_warmup', action='store_true', help='Learning rate warm-up for large batch training')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450', help='When to decay lr, as string separated by comma')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Decay rate for learning rate')
    parser.add_argument('--lr_cosine', action='store_true', help='Using cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    #parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')

    # Output & logging
    parser.add_argument('--save_freq', type=int, default=50, help='Save model every N epochs.')
    parser.add_argument('--print_freq', type=int, default=10, help='Print training progress every N steps.')
    
    args = parser.parse_args()

    args.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(args.dataset, args.model, args.lr, args.weight_decay,
               args.batch_size, args.trial)

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
        args.lr_warmup_from = 0.01
        args.lr_warmup_epochs = 10
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warmup_epochs / args.epochs)) / 2
        else:
            args.lr_warmup_to = args.lr

    # Prepare dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))
    
    # Set-up output directory
    args.save_dir = f'./output/{args.experiment_name}/{args.model_name}'
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def set_model(args):
    model = SupCEResNet(name=args.model, num_classes=args.num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    # Enable synchronized Batch Normalization
    """if args.syncBN:
        model = apex.parallel.convert_syncbn_model(model)"""

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
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
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # Warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # Compute loss
        output = model(images)
        loss = criterion(output, labels)

        # Update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        wandb.log({"train_loss": loss.item(), "top1_accuracy": acc1[0], "lr": optimizer.param_groups[0]['lr']})

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args):
    """One epoch validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # Forward
            output = model(images)
            loss = criterion(output, labels)

            # Update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    args = parse_args()

    # Build dataloaders
    print("Preparing dataloaders...")

    train_loader = set_loader(args, split="train", aug_mode=args.aug_method)
    val_loader = set_loader(args, split="val", aug_mode=None)

    print("Dataloaders ready.")

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
            "batch_size" : args.batch_size,
            "epochs" : args.epochs,
            "lr" : args.lr,
            "lr_warmup" : args.lr_warmup,
            "lr_cosine" : args.lr_cosine,
            "weight_decay" : args.weight_decay,
        },
    )

    # Training routine
    print("Initiating training...")

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # Train & validate for one epoch
        start_time = time.time()

        avg_train_loss, avg_train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        avg_val_loss, avg_val_acc = validate(val_loader, model, criterion, args)

        end_time = time.time()

        print('epoch {}, total time {:.2f}'.format(epoch, end_time - start_time))

        # Log average epoch metrics
        wandb.log({
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "avg_train_acc": avg_train_acc,
            "avg_val_acc": avg_val_acc,
            #"avg_ID_confidence": avg_ID_confidence,
            #"avg_OOD_confidence": avg_OOD_confidence,
        })

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc

        # Save model
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # Save the last model
    save_file = os.path.join(
        args.save_dir, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
