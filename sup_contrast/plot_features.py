import argparse

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import torch

from main_linear import set_loader, set_model

def parse_args():
    parser = argparse.ArgumentParser('Arguments for plotting features')

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to model from contrastive pre-training stage.')

    args = parser.parse_args()

    args.dataset = "mvip"
    args.num_classes = 20
    args.aug_dir_id = None
    args.aug_dir_ood = None
    args.image_size = 224
    args.batch_size = 1
    args.num_workers = 0

    return args

def plot_features(features, labels, title):
    # Apply t-SNE to features
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    # Plot t-SNE results & save figure
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar()
    plt.savefig(f'{title}.png')

if __name__ == '__main__':
    args = parse_args()

    # Build dataloader
    test_loader = set_loader(args, split="test", aug_mode=None)

    # Load model
    model, _, _ = set_model(args)

    # Get features & labels for whole test set & plot in single figure
    all_features = []
    all_labels = []

    for i, (images, labels) in enumerate(test_loader):
        print(f'Batch {i+1}/{len(test_loader)}')
        features = model(images)
        all_features.append(features)
        all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    plot_features(all_features, all_labels, 'Test Set Features')