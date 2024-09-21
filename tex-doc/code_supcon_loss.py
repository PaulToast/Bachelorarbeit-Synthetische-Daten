class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             ' at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Get/calculate contrastive mask for this batch (shape=[bsz, bsz])
        # mask[i][j]=1 if sample j has the same class as sample i, otherwise 0
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        # Update mask, so that self-contrast cases aren't considered as positive pairs
        self_contrast_mask = 1 - torch.eye(batch_size, dtype=torch.float32).to(device)
        mask *= self_contrast_mask
        # Also make sure there are no positive pairs with any OOD samples (negative labels)
        ood_mask = (labels < 0).float().to(device).detach()
        mask *= (1 - ood_mask @ ood_mask.T)

        # Determine whether all views or just one of each sample will be used as anchor
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

        # Compute logits (similarity scores for anchor & contrast features)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Repeat masks to match the dimensions of the logits
        mask = mask.repeat(anchor_count, contrast_count)
        self_contrast_mask = self_contrast_mask.repeat(anchor_count, contrast_count)

        # Ignore logits for OOD samples where OOD label is not the negative anchor label
        non_ood_mask = 1 - ood_mask
        valid_ood_mask = (labels == -labels.T).float().to(device).detach()
        logits *= (non_ood_mask + valid_ood_mask).repeat(anchor_count, contrast_count)

        # Transform the logits into log-probabilities
        exp_logits = torch.exp(logits) * self_contrast_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Looking only at positive pairs, compute the mean log-probabilities for each anchor
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Final loss: average negative log-probabilities over all positive pairs
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        del self_contrast_mask, ood_mask, non_ood_mask, valid_ood_mask

        return loss