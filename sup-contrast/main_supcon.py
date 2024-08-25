from __future__ import print_function

import os
import sys
import json
import argparse
import time
import math
import numpy as np

#import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.utils import save_image

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss

from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage import maximum_filter

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
        split="train",
        size=224,
        transform=None,
        repeats=1, # da-fusion: 100
        synt=None
    ):
        self.data_root = '/mnt/HDD/MVIP/sets'

        split_dir = {
            "train": "train_data",
            "val": "valid_data",
            "test": "test_data",
        }
        self.split = split_dir[split]

        self.size = size

        # Limit dataset to 20 classes from the "CarComponent" super class
        self.class_names = []

        for class_name in [f for f in os.listdir(self.data_root) if os.path.isdir(os.path.join(self.data_root, f))]:
            meta_file = open(os.path.join(self.data_root, class_name, "meta.json"))
            meta_data = json.load(meta_file)

            if "CarComponent" in meta_data['super_class']:
                self.class_names.append(class_name)

            meta_file.close()

            del self.class_names[20:]

        self.all_images, self.all_masks = self.get_all_image_paths(self.class_names, self.split)
        self.num_images = len(self.all_images)

        self.class_to_label_id = {self.class_names[i]: i for i in range(len(self.class_names))}
        self.all_labels = [self.class_to_label_id[self.all_images[i].split("/")[-6 if split == "train" else -5]] for i in range(self.num_images)]
        # Example: "/mnt/HDD/MVIP/sets/ >CLASS_NAME< /train_data/0/0/cam0/0_rgb.png"
        # Example: "/mnt/HDD/MVIP/sets/ >CLASS_NAME< /valid_data/0/cam0/0_rgb.png"

        # Shuffle dataset
        np.random.seed(0)
        shuffle_idx = np.random.permutation(self.num_images)
        self.all_images = [self.all_images[i] for i in shuffle_idx]
        self.all_masks = [self.all_masks[i] for i in shuffle_idx]
        self.all_labels = [self.all_labels[i] for i in shuffle_idx]

        if split == "train":
            self._length = self.num_images * repeats
        else:
            self._length = self.num_images

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.size, scale=(0.8, 1.)),# ratio=(1.0, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Get label
        label = self.all_labels[idx % self.num_images]

        # Get image
        image = Image.open(self.all_images[idx % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Get object mask & use maximum filter to dilate it
        mask = np.array(Image.open(self.all_masks[idx % self.num_images]).convert('L'))
        mask = Image.fromarray(maximum_filter(mask, size=32))

        # Use mask to crop image
        image = self.mask_crop(image, mask)

        return self.transform(image), label
    
    def get_all_image_paths(self, class_names, split): # Example: "/mnt/HDD/MVIP/sets/class_name/train_data/0/0/cam0/0_rgb.png"
        images = []
        masks = []

        for class_name in class_names:
            root = os.path.join(self.data_root, class_name, split)

            for set in [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]:
                if split == "train_data":
                    for orientation in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for cam in [f for f in os.listdir(os.path.join(root, set, orientation)) if os.path.isdir(os.path.join(root, set, orientation, f))]:
                            for file in os.listdir(os.path.join(root, set, orientation, cam)):
                                if file.endswith("rgb.png"):
                                    images.append(os.path.join(root, set, orientation, cam, file))
                                elif file.endswith("rgb_mask_gen.png"):
                                    masks.append(os.path.join(root, set, orientation, cam, file))
                else:
                    for cam in [f for f in os.listdir(os.path.join(root, set)) if os.path.isdir(os.path.join(root, set, f))]:
                        for file in os.listdir(os.path.join(root, set, cam)):
                            if file.endswith("rgb.png"):
                                images.append(os.path.join(root, set, cam, file))
                            elif file.endswith("rgb_mask_gen.png"):
                                masks.append(os.path.join(root, set, cam, file))
        
        return images, masks
    
    def get_mean_std(self):
        mean = torch.zeros(3)
        std = torch.zeros(3)

        for image in self.all_images:
            image = Image.open(image)
            image = self.transform(image)
            mean += torch.mean(image, dim=(1, 2))
            std += torch.std(image, dim=(1, 2))

        mean /= len(self.all_images)
        std /= len(self.all_images)

        return mean, std
    
    def mask_crop(self, image: Image, mask: Image):
        mask_box = mask.getbbox()

        # Make mask_box square without offsetting the center
        mask_box_width = mask_box[2] - mask_box[0]
        mask_box_height = mask_box[3] - mask_box[1]
        mask_box_size = max(mask_box_width, mask_box_height)
        mask_box_center_x = (mask_box[2] + mask_box[0]) // 2
        mask_box_center_y = (mask_box[3] + mask_box[1]) // 2
        mask_box = (
            mask_box_center_x - mask_box_size // 2,
            mask_box_center_y - mask_box_size // 2,
            mask_box_center_x + mask_box_size // 2,
            mask_box_center_y + mask_box_size // 2
        )

        # Crop image with mask_box
        return image.crop(mask_box)


def parse_args():
    parser = argparse.ArgumentParser('Arguments for training')

    # Dataset & model
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mvip'])
    parser.add_argument('--data_dir', type=str, default=None)

    parser.add_argument('--get_samples', action='store_true', help='Export sample images from dataset.')

    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs.')

    # Training parameters
    parser.add_argument('--method', type=str, default='SupCon', choices=['SupCon', 'SimCLR'])

    parser.add_argument('--size', type=int, default=224, help='Size for RandomResizedCrop.') #32

    parser.add_argument('--batch_size', type=int, default=16) #256
    parser.add_argument('--num_workers', type=int, default=4) #16
    parser.add_argument('--epochs', type=int, default=110) #1000

    parser.add_argument('--lr', type=float, default=0.002) #0.05
    parser.add_argument('--lr_warmup', action='store_true', help='Learning rate warm-up for large batch training.')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900', help='When to decay lr, as string seperated by ","')
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
        format(args.method, args.dataset, args.model, args.lr,
               args.weight_decay, args.batch_size, args.temp, args.trial)

    if args.lr_cosine:
        args.model_name = '{}_cosine'.format(args.model_name)

    # Learning rate warm-up for large-batch training
    if args.batch_size > 256:
        args.lr_warmup = True
    if args.lr_warmup:
        args.model_name = '{}_warm'.format(args.model_name)
        args.lr_warmup_from = 0.0001
        args.lr_warmup_epochs = 1
        if args.lr_cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.lr_warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.lr_warm_epochs / args.epochs)) / 2
        else:
            args.lr_warmup_to = args.lr

    args.logging_dir = os.path.join(args.logging_dir, args.model_name)
    if not os.path.isdir(args.logging_dir):
        os.makedirs(args.logging_dir)

    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    return args


# Construct data loader
def set_loader(args, split="train"):
    mean_std = {
        'cifar10': ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        'cifar100': ([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        'mvip': ([0.4213, 0.4252, 0.4242], [0.1955, 0.1923, 0.1912]),
    }

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.8, 1.)),# ratio=(1.0, 1.0)),
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
        dataset = MVIPDataset(split=split,
                              size=args.size,
                              transform=TwoCropTransform(transform))

    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=sampler)
    
    """if args.get_samples:
        print("Exporting sample images...")
        for i in range(4):
            images, label = train_dataset[i]
            img1, img2 = images
            class_name = train_dataset.class_names[label]
            save_image(img1, f"sample_{i}_{class_name}_0.png")
            save_image(img2, f"sample_{i}_{class_name}_1.png")"""

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

        # Update metric & log
        losses.update(loss.item(), bsz)
        wandb.log({"train_loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})

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

def validate(val_loader, model, criterion, args):
    """One epoch validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        start_time = time.time()
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
            batch_time.update(time.time() - start_time)

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

    print("Dataloaders ready.")

    # Build model and criterion
    model, criterion = set_model(args)

    # Build optimizer
    optimizer = set_optimizer(args, model)

    # Init W&B logging
    run = wandb.init(
        project='SupCon', #project=f"SupCon-v{args.trial}",
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

        # Train for one epoch
        start_time = time.time()

        avg_train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        avg_val_loss = validate(val_loader, model, criterion, args)

        end_time = time.time()

        print('epoch {}, total time {:.2f}, average train loss: {}, average validation loss: {}'.format(
            epoch, end_time - start_time, avg_train_loss, avg_val_loss))

        # Log average epoch loss
        wandb.log({
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
        })

        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # Save the last model
    save_file = os.path.join(args.save_dir, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)


if __name__ == '__main__':
    main()
