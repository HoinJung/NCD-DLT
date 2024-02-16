import numpy as np
import torch
from torch.utils.data import Dataset, Subset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        # try : 
        #     self.labeled_target = list(set(labelled_dataset.targets))
        # except : 
        #     self.labeled_target  = list(set(labelled_dataset.data.target.values))

        self.unlabelled_dataset = unlabelled_dataset

        # try : 
        #     self.unlabeled_target = list(set(unlabelled_dataset.targets))
        # except : 
        #     self.unlabeled_target  = list(set(unlabelled_dataset.data.target.values))
        # self.targets = list(set(self.unlabeled_target + self.labeled_target))
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)



# class OnlineDataset(Dataset):

#     def __init__(self, unlabelled_dataset,total_round):

        
#         self.unlabelled_dataset = unlabelled_dataset
#         self.target_transform = None
#         self.total_round = total_round

#     def __getitem__(self, item):
        
#         indices = torch.randperm(len(self.unlabelled_dataset))
#         b = len(unlabelled_dataset)//self.total_round
#         subset_indices = [indices[j:j+b] for j in range(0, len(unlabelled_dataset), b)]
#         if len(subset_indices)!=self.total_round:
#             subset_indices = subset_indices[:-1] 
#         random.shuffle(subset_indices)
#         subsets = [Subset(unlabelled_dataset, indices) for indices in subset_indices]
        

#         if item < len(self.labelled_dataset):
#             img, label, uq_idx = self.labelled_dataset[item]
#             labeled_or_not = 1

#         else:

#             img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
#             labeled_or_not = 0


#         return img, label, uq_idx, np.array([labeled_or_not])

#     def __len__(self):
#         return len(self.unlabelled_dataset) + len(self.labelled_dataset)
