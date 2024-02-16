
import copy
import numpy as np
import torch
from torch.nn import functional as F
import copy
import itertools

def hamming_distance(s1, s2):
    """Compute the Hamming distance between two strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def find_largest_neighbor(hash_code, unique_hash_codes, counts, hash_to_index,threshold, max_hamming=1):
    while max_hamming <= len(hash_code):
        neighbors = [code for i, code in enumerate(unique_hash_codes) if hamming_distance(hash_code, code) == max_hamming]
        if not neighbors:
            max_hamming += 1
            continue        
        # Find the neighbor with the maximum count
        max_neighbor = max(neighbors, key=lambda x: counts[hash_to_index[tuple(x)]])
        return max_neighbor
    return None

def merge_based_on_confidence(feats, threshold, binarization_threshold=0.00, confidence_interval=0.3):
    
    feats_hash = (feats > binarization_threshold).astype(int)
    unique_hash_codes, inverse, counts = np.unique(feats_hash, axis=0, return_inverse=True, return_counts=True)
    hash_to_index = {tuple(code): index for index, code in enumerate(unique_hash_codes)}
    
        
    for feat in feats:
        
        hash = (feat > binarization_threshold).astype(int)
        feat_index = np.where((unique_hash_codes == hash).all(axis=1))[0][0]
        if counts[feat_index] > threshold:
            continue  
        confidence_feat = np.where((binarization_threshold - confidence_interval<=feat) & (feat<=binarization_threshold+confidence_interval),feat,0)
        low_confidence_indices = np.where((binarization_threshold - confidence_interval<=feat) & (feat<=binarization_threshold+confidence_interval))[0]
        subsets = list(itertools.chain.from_iterable(itertools.combinations(low_confidence_indices, r) for r in range(len(low_confidence_indices) + 1)))
        L = len(hash)
        soft_hamming_similarities = []
        if len(low_confidence_indices)==0:
            continue
        for subset in subsets:
            candidate = copy.deepcopy(hash)
            distance = 0
            for i in subset:
                candidate[i] = 1-hash[i]
                distance += np.abs(confidence_feat)[i]
            index = np.where((unique_hash_codes == candidate).all(axis=1))[0]
            soft_hamming_similarity = L*confidence_interval - distance
            soft_hamming_similarities.append(soft_hamming_similarity)
        softmax_scores = softmax(np.array(soft_hamming_similarities))
        weights = []


        for subset, softmax_score in zip(subsets, softmax_scores):
            candidate = copy.deepcopy(hash)
            for i in subset:
                candidate[i] = 1-hash[i]
            indices = np.where((unique_hash_codes == candidate).all(axis=1))
            if indices[0].size == 0:
                continue
            index = indices[0][0]  # Get the first matching hash code's index
            weight = counts[index] * softmax_score
            weights.append(weight)
        

        # Pick the candidate with the largest weight
        max_weight_index = np.argmax(weights)
        max_weight_candidate = subsets[max_weight_index]
        selected_candidate = copy.deepcopy(hash)
        for i in max_weight_candidate:  
            selected_candidate[i] = 1 - hash[i]

        if tuple(selected_candidate) in hash_to_index:
            target_index = np.where((unique_hash_codes == hash).all(axis=1))[0][0]
            inverse[inverse == target_index] = hash_to_index[tuple(selected_candidate)]
        else:
            # Find the largest neighbor
            max_neighbor = find_largest_neighbor(hash, unique_hash_codes, counts,hash_to_index,threshold)
            if max_neighbor is not None:
                inverse[inverse == np.where((unique_hash_codes == hash).all(axis=1))[0][0]] = hash_to_index[tuple(max_neighbor)]
    # The merged hash codes are just the unique hash codes after considering the merging
    merged_hash_codes = unique_hash_codes
    
    return merged_hash_codes, inverse



class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, device,temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device
    def forward(self, features,labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = self.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
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

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def info_nce_logits(features, args):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(args.device)

    logits = logits / args.temperature
    return logits, labels


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

