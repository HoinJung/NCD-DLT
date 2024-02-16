# NCD-DLT
## Introduction
NCD-DLT aims to discover novel classes in a challenging condition, where new unlabeled data keeps arriving dynamically within the long-tailed distribution.




## Install
### Requirements
  ```
cuda=11.6.1
numpy=1.23.5
pandas=1.5.3
python=3.10.9
torch=2.0.1
  ```
### Clone Repository
```
git clone https://github.com/hin1115/NCD-DLT.git
```

## Pretrained DINO
1. Download the ViT-B/16 backbone checkpoint from this repository. [Link](https://github.com/facebookresearch/dino?tab=readme-ov-file) 
2. Locate the DINO's pretrained weight 'pretrained_models' directory.


## Run code
- ```--greedy```: Use greedy hash regulazation loss.
- ```--double```: Use double hashing in the projection head.
- ```--pseudo```: Use the Hamming hash graph merging algorithm as a pseudo label with supervised contrastive learning.
- ```--threshold_size```: Node size threshold for graph merging.
- ```--threshold_confidence```: Confidence threshold for graph mering.
- Post-processing with graph merging will be applied automatically. Both the naive accuracy and post-processed accuracy will displayed together.
- 
```
python NCD-DLT.py --gpu_id 0  --init_epochs 50 --epochs 10  --dataset_name cifar100 --greedy --double --pseudo --lam 10 --threshold_size 10 --threshold_confidence 0.2 
```


## Test
During the training, the evaluation is conducted at every epoch in the initial stage and the last epoch of every dynamic stage.

## Citation
TBD

## LICENSE
MIT License


