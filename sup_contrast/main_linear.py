from __future__ import print_function

import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier

from mvip_dataset import MVIPDatasetSupCon

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

    parser.add_argument('--output_name', type=str, default=None, help='Output directory name for run.')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs.')

    # Data
    parser.add_argument('--dataset', type=str, default='mvip', choices=['cifar10', 'cifar100', 'mvip'])
    parser.add_argument('--data_dir', type=str, default=None)

    parser.add_argument('--aug_mode',
                        type=str,
                        default=None,
                        choices=[None, 'with_id'],
                        help=("Augmentation mode for contrastive training. ",
                              "If None, no augmentations are used. ",
                              "If 'with_id', normal augmentations are added for training. ",
                              "OOD augmentations are not supported for linear classification stage, ",
                              "but are used for validating OOD detection."))
    parser.add_argument('--aug_output_name',
                        type=str,
                        default="mvip-v9-final",
                        help="DA-Fusion output name for the augmentations.")
    parser.add_argument('--aug_name_id',
                        type=str,
                        default="aug=0.2_ex=16_num=4_g=15",
                        help="Name of the subfolder containing the in-distribution augmentations.")
    parser.add_argument('--aug_name_ood',
                        type=str,
                        default="aug=0.5_ex=16_num=4_g=15",
                        help="Name of the subfolder containing the out-of-distribution augmentations.")

    # Training
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--ckpt', type=str, default='', help='Path to model from contrastive pre-training stage.')

    parser.add_argument('--image_size', type=int, default=224, help='Size for RandomResizedCrop.') #32

    parser.add_argument('--epochs', type=int, default=50) #100
    parser.add_argument('--batch_size', type=int, default=16) #256
    parser.add_argument('--num_workers', type=int, default=4) #16

    parser.add_argument('--lr', type=float, default=0.001) #0.1
    parser.add_argument('--lr_warmup', action='store_true', help='Learning rate warm-up for large batch training.')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='When to decay lr, as string separated by comma.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_cosine', action='store_true', help='Using cosine annealing.')

    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)

    # Output & logging
    parser.add_argument('--save_freq', type=int, default=10, help='Save model every N epochs.') #50
    parser.add_argument('--print_freq', type=int, default=10, help='Print training progress every N steps.')
    
    args = parser.parse_args()

    args.ckpt = os.path.abspath(args.ckpt)

    args.model_name = '{}_{}_trial={}_lr={}_decay={}_bsz={}'.\
        format(args.dataset, args.model, args.trial, args.lr, args.weight_decay, args.batch_size)

    # Set-up learning rate
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    if args.lr_cosine:
        args.model_name = '{}_cosine'.format(args.model_name)

    if args.lr_warmup:
        args.model_name = '{}_warmup'.format(args.model_name)
        args.lr_warmup_from = 0.01
        args.lr_warm_epochs = 10
        if args.cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
        else:
            args.lr_warmup_to = args.lr

    # Prepare datasets
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'mvip':
        args.num_classes = 20

        if args.aug_name_id is not None:
            args.aug_dir_id = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da_fusion/output/{args.aug_output_name}/{args.aug_name_id}'
            ))
        else:
            args.aug_dir_id = None

        if args.aug_name_ood is not None:
            args.aug_dir_ood = os.path.abspath(os.path.join(
                os.path.dirname( __file__ ), '..', f'da_fusion/output/{args.aug_output_name}/{args.aug_name_ood}'
            ))
        else:
            args.aug_dir_ood = None
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    # Set-up output directory
    args.save_dir = os.path.abspath(f'output/{args.output_name}/trial={args.trial}')
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


def set_loader(args, split="train", aug_mode=None):
    mean_std = {
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        'mvip': ([0.4213, 0.4252, 0.4242], [0.1955, 0.1923, 0.1912]),
    }

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.image_size, scale=(0.8, 1.)), # ratio=(1.0, 1.0))
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15.0),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[args.dataset][0], std=mean_std[args.dataset][1]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_std[args.dataset][0], std=mean_std[args.dataset][1]),
        ])

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root=args.data_dir,
                                   train=(split == "train"),
                                   transform=transform,
                                   download=True)
    elif args.dataset == "cifar100":
        dataset = datasets.CIFAR100(root=args.data_dir,
                                    train=(split == "train"),
                                    transform=transform,
                                    download=True)
    elif args.dataset == "mvip":
        dataset = MVIPDatasetSupCon(split=split,
                              aug_mode=aug_mode,
                              aug_dir_id=args.aug_dir_id,
                              aug_dir_ood=args.aug_dir_ood,
                              transform=transform)

    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=sampler)

    return dataloader


def set_model(args):
    model = SupConResNet(name=args.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=args.model, num_classes=args.num_classes)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    """One epoch training"""
    model.eval()
    classifier.train()

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
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
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


def validate(val_loader, ood_loader, model, classifier, criterion, args):
    """One epoch validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    id_confidences = AverageMeter()
    ood_confidences = AverageMeter()

    with torch.no_grad():
        # Loss & accuracy validation
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # Forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # Update metrics
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # Log in-distribution confidence
            id_probabilities = F.softmax(output, dim=1)
            id_confidence = torch.max(id_probabilities, dim=1).values.mean().item()
            id_confidences.update(id_confidence, 1)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if idx % args.print_freq == 0:
                print('Generalization Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
        
        # OOD validation
        end = time.time()
        for idx, (ood_images, ood_labels) in enumerate(ood_loader):
            ood_images = ood_images.float().cuda()
            ood_labels = ood_labels.cuda()

            ood_output = classifier(model.encoder(ood_images))

            # Log out-of-distribution confidence
            ood_probabilities = F.softmax(ood_output, dim=1)
            ood_confidence = torch.max(ood_probabilities, dim=1).values.mean().item()
            ood_confidences.update(ood_confidence, 1)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if idx % args.print_freq == 0:
                print('OOD Detection Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'OOD confidence {conf.val:.4f} ({conf.avg:.4f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       conf=ood_confidences))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg, id_confidences.avg, ood_confidences.avg


def main():
    args = parse_args()

    # Build dataloaders
    print("Preparing dataloaders...")

    train_loader = set_loader(args, split="train", aug_mode=args.aug_mode)
    val_loader = set_loader(args, split="val", aug_mode=None)
    ood_loader = set_loader(args, split="val", aug_mode="ood_only")

    print("Dataloaders ready.")

    # Build model and criterion
    model, classifier, criterion = set_model(args)

    # Build optimizer
    optimizer = set_optimizer(args, classifier)

    # Init W&B logging
    run = wandb.init(
        project=args.output_name,
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

        avg_train_loss, avg_train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, args)
        avg_val_loss, avg_val_acc, avg_id_confidence, avg_ood_confidence = validate(val_loader, ood_loader, model, classifier, criterion, args)
        
        end_time = time.time()
        
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
        
        print('Epoch {}, total time {:.2f}, avg train acc: {:.2f}, avg val acc: {:.2f}, avg ID confidence: {:.2f}, avg OOD confidence: {:.2f}'.format(
            epoch, end_time - start_time, avg_train_acc, avg_val_acc, avg_id_confidence, avg_ood_confidence))
        
        # Log average epoch metrics
        wandb.log({
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "avg_train_acc": avg_train_acc,
            "avg_val_acc": avg_val_acc,
            "avg_ID_confidence": avg_id_confidence,
            "avg_OOD_confidence": avg_ood_confidence,
        })

        # Save model
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)
    
    # Save the last model
    save_file = os.path.join(args.save_dir, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)

    print('Best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
