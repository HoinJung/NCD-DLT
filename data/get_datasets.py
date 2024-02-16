from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.tiny_imagenet import get_tiny_200_datasets

from data.cifar import lc_get_cifar_100_datasets,lc_get_cifar_10_datasets
from data.tiny_imagenet import lc_get_tiny_200_datasets

from copy import deepcopy
import pickle
import os

from config import osr_split_dir


get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'tiny_imagenet': get_tiny_200_datasets,
}


lc_get_dataset_funcs = {
    'cifar10': lc_get_cifar_10_datasets,
    'cifar100': lc_get_cifar_100_datasets,
    'tiny_imagenet': lc_get_tiny_200_datasets,
}



class TransformTwice:
    # two different random transform with one image
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2
 
    
def get_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    if args.lc:
        lc_get_dataset_f = lc_get_dataset_funcs[dataset_name]
        datasets = lc_get_dataset_f(args,train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)
    else: 
        get_dataset_f = get_dataset_funcs[dataset_name]
        datasets = get_dataset_f(args,train_transform=train_transform, test_transform=test_transform,
                        train_classes=args.train_classes,
                        prop_train_labels=args.prop_train_labels,
                        split_train_val=False)
    
    
    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    
    unlabelled_train_examples_test.transform = train_transform
    labelled_dataset = deepcopy(datasets['train_labelled'])
    labelled_dataset.transform = train_transform

    
    online_list = []
    if args.online:
        online_subsets = deepcopy(datasets['online'])
        for subset in online_subsets:
            subset.transform = train_transform
            online_list.append(subset)


    return train_dataset , test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset,online_list


def get_class_splits(args):

    
    use_ssb_splits = True
    
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)
        

    elif args.dataset_name == 'tiny_imagenet':

        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)
        

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']

    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)


    elif args.dataset_name == 'scars':
        args.image_size = 224

        if use_ssb_splits:
            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:
            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)
    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)
            
            
    elif args.dataset_name == 'chinese_traffic_signs':

        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    else:

        raise NotImplementedError

    return args