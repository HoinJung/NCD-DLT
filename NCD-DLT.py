import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import random
import copy
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
import vision_transformer as vits
from project_utils.general_utils import str2bool
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from tqdm import tqdm
from torch.nn import functional as F
from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import dino_pretrain_path
from utils import *


def train(args, projection_head, model, train_loader, online_dataset, test_loader,labelled_dataset):
    global suffix
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    device = args.device
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.init_epochs,
            eta_min=args.lr * 1e-3,
        )

    sup_con_crit = SupConLoss(device)
    sup_model_path =  'checkpoints/dynamic_sup_'+suffix

    try :
        checkpoint_model = torch.load(sup_model_path+'_model.pt')
        checkpoint_head = torch.load(sup_model_path + '_proj_head.pt')    
        model.load_state_dict(checkpoint_model)
        projection_head.load_state_dict(checkpoint_head)
        print("Load pretrained model")
    except :
        print("initial training")
        for epoch in tqdm(range(args.init_epochs)):
            projection_head.train()
            model.train()
            for _, batch in enumerate(train_loader):
                images, class_labels, _ = batch
                class_labels = class_labels.to(device)
                images = torch.cat(images, dim=0).to(device)
                features = model(images)
                features, hash_features, variance_features = projection_head(features)
                features = torch.nn.functional.normalize(features, dim=-1)
                
                f1, f2 = [f for f in features.chunk(2)]
                
                sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                sup_con_labels = class_labels
                sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)

                if args.greedy:
                    f1, f2 = [f for f in features.chunk(2)]
                    h1, h2 = [f for f in hash_features.chunk(2)]
                    
                    h1 = torch.where(h1>0,1.0,-1.0)
                    h2 = torch.where(h2>0,1.0,-1.0)
                
                    target_b = F.cosine_similarity(h1, h2)
                    target_x = F.cosine_similarity(f1, f2)
                    hash_loss = F.mse_loss(target_b, target_x)
                    reg_loss = (1 - torch.abs(hash_features)).mean()
                    loss = sup_con_loss + reg_loss + hash_loss
                else : 
                    reg_loss = (1 - torch.abs(hash_features)).mean()
                    loss = sup_con_loss+ reg_loss 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                (all_acc_test, old_acc_test, new_acc_test),(all_acc_test2, old_acc_test2, new_acc_test2),(all_acc_test3, old_acc_test3, new_acc_test3), (all_acc_test4, old_acc_test4, new_acc_test4),(origin_n_clusters,n_clusters) = test_on_the_fly(model, projection_head, test_loader,
                                                                    epoch=epoch, save_name='Test ACC',
                                                                    args=args,phase='init',num_round = None)
            # Step schedule
            exp_lr_scheduler.step()
            
        
        torch.save(model.state_dict(),sup_model_path+'_model.pt')
        torch.save(projection_head.state_dict(), sup_model_path + '_proj_head.pt')
        checkpoint_model = torch.load(sup_model_path+'_model.pt')
        checkpoint_head = torch.load(sup_model_path + '_proj_head.pt')    
        model.load_state_dict(checkpoint_model)
        projection_head.load_state_dict(checkpoint_head)
    
    old_model = copy.deepcopy(model)
    old_head = copy.deepcopy(projection_head)
    old_model.eval()
    old_head.eval()
    for param in old_model.parameters():
        param.requires_grad = False
    for param in old_head.parameters():
        param.requires_grad = False
    print("Start Dynamic Training")
    for t in tqdm(range(args.round)):
        optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3)
        
        online_loader = DataLoader(online_dataset[t],  batch_size=args.batch_size, 
                                shuffle=True, drop_last=False)
        
        for epoch in tqdm(range(args.epochs)):
            projection_head.train()
            model.train()
            
            for _, batch in enumerate(online_loader):
                images, class_labels, _ = batch
                class_labels = class_labels.to(device)
                images = torch.cat(images, dim=0).to(device)
                
                features = model(images)
                features, hash_features, _ = projection_head(features)
                features = torch.nn.functional.normalize(features, dim=-1)
                
                contrastive_logits, contrastive_labels = info_nce_logits(features=features, args=args)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                temp_train_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=len(class_labels), shuffle=True, drop_last=True)
                old_images, old_class_labels, _ = next(iter(temp_train_loader))
                
                old_class_labels = old_class_labels.to(device)
                old_images = torch.cat(old_images, dim=0).to(device)
                
                old_features_old_model = old_model(old_images)
                old_features_old_model, *other = old_head(old_features_old_model)
                old_features_old_model = torch.nn.functional.normalize(old_features_old_model, dim=-1)
                old_feature_cur_model = model(old_images)
                old_feature_cur_model, *other = projection_head(old_feature_cur_model)
                old_feature_cur_model = torch.nn.functional.normalize(old_feature_cur_model, dim=-1)
        
                distill_loss = torch.nn.CosineEmbeddingLoss()(old_feature_cur_model, old_features_old_model, \
                        torch.ones(images.shape[0]).to(device))
                unsup_distill_loss=0
                
                if t>=1:
                    prev_model.eval()
                    prev_head.eval()
                    features_prev_model = prev_model(images)
                    features_prev_model, _, _ = prev_head(features_prev_model)
                    features_prev_model = torch.nn.functional.normalize(features_prev_model, dim=-1)
                    unsup_distill_loss = torch.nn.CosineEmbeddingLoss()(features, features_prev_model, \
                            torch.ones(images.shape[0]).to(device))
                if args.pseudo:
                    merged_hash_codes, new_preds = merge_based_on_confidence(hash_features.cpu().detach().numpy(),args.threshold_size,args.threshold_confidence)
                    new_preds = torch.tensor(new_preds).to(device)
                    f1, f2 = [f for f in features.chunk(2)]
                    sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                    l1, l2 = [f for f in new_preds.chunk(2)]
                    sup_con_loss = sup_con_crit(sup_con_feats, labels=l1)

                if args.greedy:
                    f1, f2 = [f for f in features.chunk(2)]
                    h1, h2 = [f for f in hash_features.chunk(2)]
                    
                    h1 = torch.where(h1>0,1.0,-1.0)
                    h2 = torch.where(h2>0,1.0,-1.0)
                    h = torch.where(hash_features>0,1.0,-1.0)
                
                    target_b = F.cosine_similarity(h1, h2)
                    target_x = F.cosine_similarity(f1, f2)
                
                    hash_loss = F.mse_loss(target_b, target_x)
                    reg_loss = (1 - torch.abs(hash_features)).mean()
                    loss = contrastive_loss + hash_loss + reg_loss + (distill_loss + unsup_distill_loss) * args.lam
                    
                else : 
                    reg_loss = (1 - torch.abs(hash_features)).mean()
                    loss = contrastive_loss + reg_loss + (distill_loss + unsup_distill_loss) * args.lam
                if args.pseudo:
                    loss += sup_con_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
    
            exp_lr_scheduler.step()

        prev_model = copy.deepcopy(model)
        prev_head = copy.deepcopy(projection_head)
        
        for param in prev_model.parameters():
            param.requires_grad = False
        for param in prev_head.parameters():
            param.requires_grad = False
        with torch.no_grad():
            (all_acc_test, old_acc_test, new_acc_test),(all_acc_test2, old_acc_test2, new_acc_test2),(all_acc_test3, old_acc_test3, new_acc_test3), (all_acc_test4, old_acc_test4, new_acc_test4),(origin_n_clusters,n_clusters) = test_on_the_fly(model, projection_head, test_loader,
                                                                epoch=epoch, save_name='Test ACC',
                                                                args=args,phase='online',num_round = t)

def test_on_the_fly(model, projection_head, test_loader,
                epoch, save_name,
                args,phase,num_round):

    model.eval()
    projection_head.eval()
    all_feats = []
    all_projectction_feats = []
    all_hash_feats = []
    targets = np.array([])
    mask = np.array([])
    
    
    for batch_idx, (images, label, _) in enumerate(test_loader):

        images = images.to(args.device)
        feats = model(images)
        projection_feats, hash_feature, _ = projection_head(feats)
        all_hash_feats.append(hash_feature.cpu().numpy())
        all_projectction_feats.append(projection_feats.cpu().numpy())
        feats = torch.nn.functional.normalize(feats, dim=-1)[:, :]
        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))
    
    all_hash_feats = np.concatenate(all_hash_feats)
    all_feats = np.concatenate(all_feats)
    all_projectction_feats = np.concatenate(all_projectction_feats)
    
    feats_hash = torch.Tensor(all_hash_feats > 0).float().tolist()
    preds = []
    hash_dict = []
    for feat in feats_hash:
        if not feat in hash_dict:
            hash_dict.append(feat)
        preds.append(hash_dict.index(feat))
    preds = np.array(preds)
    global suffix
    
    threshold = args.threshold_size
    confidence_interval =args.threshold_confidence
    origin_n_clusters = len(np.unique(preds))
    

    merged_hash_codes, new_preds = merge_based_on_confidence(all_hash_feats,threshold,confidence_interval)
    n_clusters = len(np.unique(new_preds))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_preds = kmeans.fit_predict(all_feats)
    print('best_n_clusters:', n_clusters)
    (all_acc, old_acc, new_acc), (all_acc2, old_acc2, new_acc2)  = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name='Hash',
                                                    print_output=True,num_round = num_round)
    _,_  = log_accs_from_preds(y_true=targets, y_pred=new_preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name='Merged Hash',
                                                    print_output=True,num_round = num_round)
    (all_acc3, old_acc3, new_acc3), (all_acc4, old_acc4, new_acc4)  = log_accs_from_preds(y_true=targets, y_pred=kmeans_preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name='KMeans',
                                                    print_output=True,num_round = num_round)
    
    
    return (all_acc, old_acc, new_acc),(all_acc2, old_acc2, new_acc2),(all_acc3, old_acc3, new_acc3), (all_acc4, old_acc4, new_acc4),(origin_n_clusters,n_clusters)
    


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Configuration from OCD
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--transform', type=str, default='imagenet')
    
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--code_dim', default=12, type=int)
    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)

    # Configuration for NCD-DLT
    parser.add_argument('--lam', type=float, default=3.0)
    parser.add_argument('--round', default=20, type=int)
    parser.add_argument('--imb_ratio', default=100, type=int)
    parser.add_argument('--init_epochs', default=50, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lc',default=False, action=argparse.BooleanOptionalAction,help='long-tailed classification')
    parser.add_argument('--greedy',default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--double',default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--online',default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--pseudo',default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--threshold_size', default=10, type=int)
    parser.add_argument('--threshold_confidence', type=float, default=0.2)
    
    seed_torch(0)
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu_id}')
    args = get_class_splits(args)
    if args.dataset_name == 'scars':
        args.init_epochs = 100
    
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    
    suffix = f'{args.model_name}_{args.dataset_name}_code.{args.code_dim}_lc.{args.lc}_rho.{args.imb_ratio}_init_epoch.{args.init_epochs}_lr.{args.lr}_lam.{args.lam}_greedy.{args.greedy}_double.{args.double}'
    prefix = ''

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':
        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path
        model = vits.__dict__['vit_base']()
        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)
        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
        model.to(args.device)
        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.mlp_out_dim = None
        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False
        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    else:
        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    _, test_dataset, _, _, labelled_dataset, online_dataset = get_datasets(args.dataset_name,
                                                                                            train_transform,
                                                                                            test_transform,
                                                                                            args)
    # --------------------
    # DATALOADERS
    # --------------------
    labelled_train_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=args.batch_size, 
                                shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['HASHHead'](args, in_dim=args.feat_dim,
                                out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, code_dim=args.code_dim, class_num=args.num_labeled_classes)
    projection_head.to(args.device)

    # ----------------------
    # TRAIN
    # ----------------------
    train(args, projection_head, model, labelled_train_loader, online_dataset, test_loader,labelled_dataset)