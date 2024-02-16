import torchvision
from torchvision.datasets import ImageNet
import numpy as np

import os

from copy import deepcopy
from data.data_utils import subsample_instances
from config import tiny_imagenet_root
import random
from collections import Counter


def print_dataset_info(dataset):
    # Calculate and print the class distribution in the dataset
    unique_classes, class_counts = np.unique(dataset.targets, return_counts=True)
    total_samples = len(dataset.targets)

    print("Class Distribution in the Subsampled Dataset:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"Class {cls}: {count} samples",end='\t')

    # Print the total number of samples
    print(f"\nTotal number of samples in the dataset: {total_samples}")


class CustomImageNet(ImageNet):
    def __init__(self, *args, **kwargs):
        super(CustomImageNet, self).__init__(*args, **kwargs)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):
        super(ImageNetBase, self).__init__(root, transform)
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        label = self.targets[item]
        return img, label, uq_idx
    def __len__(self):
        return len(self.targets)

class lc_ImageNetBase(torchvision.datasets.ImageFolder):
    def __init__(self, imb_ratio, *args, **kwargs):
        super(lc_ImageNetBase, self).__init__( *args, **kwargs)
        
        # Calculate the frequency of each class
        class_counts = Counter(self.targets)
        num_classes = len(class_counts)
        
        random_indices = list(range(num_classes))
        random.shuffle(random_indices)
        class_to_random_index = {cls: random_indices[i] for i, cls in enumerate(class_counts)}
        self.sample_per_class = {cls: int(count * ((1/imb_ratio) ** (class_to_random_index[cls] / (num_classes-1)))) for cls, count in class_counts.items()}
        
        # Create a downsampled dataset
        self.downsampled_indices = self._downsample_indices()

        # Update the targets and data accordingly
        self.targets = [self.targets[i] for i in self.downsampled_indices]
        # self.data = self.data[self.downsampled_indices]
        self.samples = [self.samples[i] for i in self.downsampled_indices]
        self.uq_idxs = np.array(range(len(self)))

    def _downsample_indices(self):
        indices = []
        class_indices = {cls: np.where(np.array(self.targets) == cls)[0] for cls in range(200)}

        for cls, count in self.sample_per_class.items():
            count = max(count, 1)
            cls_indices = class_indices[cls]
            np.random.shuffle(cls_indices)
            indices.extend(cls_indices[:count])

        return np.array(indices)

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        label = self.targets[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_tiny_200_datasets(args,train_transform, test_transform, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)
    # Init entire training set    
    whole_training_set = ImageNetBase(root=os.path.join(tiny_imagenet_root, 'train'), transform=train_transform)
    
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))


    # Create random incremental stages
    b = len(unlabelled_indices)//args.round
    unlabel_list = list(unlabelled_indices)
    random.shuffle(unlabel_list)
    unlabelled_indices = set(unlabel_list)
    subset_indices = [unlabel_list[j:j+b] for j in range(0, len(unlabelled_indices), b)]
    if len(subset_indices)!=args.round:
        subset_indices = subset_indices[:-1] 
    random.shuffle(subset_indices)
    online_train_dataset_unlabelled_subsets = [subsample_dataset(deepcopy(whole_training_set), \
        np.array(list(indices))) for indices in subset_indices]

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(tiny_imagenet_root, 'val'), transform=test_transform)
    

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    # print_dataset_info(train_dataset_labelled)
    # print_dataset_info(train_dataset_unlabelled)
    # print_dataset_info(test_dataset)
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'online':online_train_dataset_unlabelled_subsets
    }

    return all_datasets


def lc_get_tiny_200_datasets(args,train_transform, test_transform, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)
    
    whole_training_set = lc_ImageNetBase(imb_ratio=args.imb_ratio, root=os.path.join(tiny_imagenet_root, 'train'), transform=train_transform)
    
    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))


    # Create random incremental stages
    b = len(unlabelled_indices)//args.round
    unlabel_list = list(unlabelled_indices)
    random.shuffle(unlabel_list)
    unlabelled_indices = set(unlabel_list)
    subset_indices = [unlabel_list[j:j+b] for j in range(0, len(unlabelled_indices), b)]
    if len(subset_indices)!=args.round:
        subset_indices = subset_indices[:-1] 
    random.shuffle(subset_indices)
    online_train_dataset_unlabelled_subsets = [subsample_dataset(deepcopy(whole_training_set), \
        np.array(list(indices))) for indices in subset_indices]

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(tiny_imagenet_root, 'val'), transform=test_transform)
    

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    print_dataset_info(train_dataset_labelled)
    print_dataset_info(train_dataset_unlabelled)
    print_dataset_info(test_dataset)
    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'online':online_train_dataset_unlabelled_subsets
    }

    return all_datasets



if __name__ == '__main__':

    x = get_tiny_200_datasets(None, None, split_train_val=False,
                               train_classes=range(50), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')