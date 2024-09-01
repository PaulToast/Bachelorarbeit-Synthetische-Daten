"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz],
                mask_{i,j}=1 if sample j has the same class as sample i.
                Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             ' at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        """# Remove OOD samples if necessary (limit to 20% of batch size)
        original_batch_size = features.shape[0]

        if labels is not None:
            ood_count = torch.sum(labels == -1)
            ood_limit = int(0.2 * original_batch_size)
            if ood_count > ood_limit:
                num_samples_to_remove = ood_count - ood_limit
                ood_indices = torch.where(labels == -1)[0]
                indices_to_remove = ood_indices[:num_samples_to_remove]

                features = torch.index_select(features, 0,
                                              torch.tensor([i for i in range(original_batch_size) if i not in indices_to_remove]).to(device))
                labels = torch.index_select(labels, 0,
                                            torch.tensor([i for i in range(original_batch_size) if i not in indices_to_remove]).to(device))"""

        new_batch_size = features.shape[0]

        # Get/calculate contrastive mask for this batch (shape=[bsz, bsz])
        # mask_{i,j}=1 if sample j has the same class as sample i, otherwise 0
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(new_batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != new_batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # Make sure there are no positive pairs with any OOD samples (label=-1)
        ood_mask = (labels < 0).float().to(device).detach()
        mask *= (1 - ood_mask @ ood_mask.T)

        # Determine whether all views or just one of each sample will be used as the anchor
        contrast_count = features.shape[1] # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all': # All features used as anchor & contrast
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # Compute logits, representing the similarity scores between anchor & contrast features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Update contrastive mask to filter out self-contrast cases
        mask = mask.repeat(anchor_count, contrast_count) # Repeat mask to match the dimensions of the logits
        self_contrast_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(new_batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * self_contrast_mask

        # Also ignore logits for OOD samples where OOD label is not the negative anchor label
        non_ood_mask = 1 - ood_mask
        valid_ood_mask = (labels == -labels.T).float().to(device).detach()
        logits *= (non_ood_mask + valid_ood_mask).repeat(anchor_count, contrast_count)

        # Transform the logits into log-probabilities (of a pair being *more* similar than any other pair)
        # -> We don't just want to maximize similarity for positive pairs,
        #    but maximize the likelihood of them being *more* similar than negative pairs
        exp_logits = torch.exp(logits) * self_contrast_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Look only at the positive pairs & compute the mean log-probabilities for each anchor
        # Modified to handle edge cases when there is no positive pair for an anchor point
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Final loss calculation, averaging the negative log-probabilities over all positive pairs
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, new_batch_size).mean()

        """# Scale loss depending on batch size reduction
        loss *= original_batch_size / new_batch_size"""

        del ood_mask, non_ood_mask, valid_ood_mask

        return loss