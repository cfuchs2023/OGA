# -*- coding: utf-8 -*-
from tqdm import tqdm

import torch
import torch.nn.functional as F
import os
import clip
import json

def clip_classifier(classnames, template, clip_model, reduce='mean', gpt=False, wordnet_dict=None):
    with torch.no_grad():
        clip_weights = []
        if wordnet_dict is not None:
            indices = []
            i = 0
            for classname in classnames:
                allnames = [classname] + wordnet_dict[classname]
                for name in allnames:
                   
                    # Tokenize the prompts
                    name = name.replace('_', ' ')
                    
                    texts = [t.format(name) for t in template]
                    texts = clip.tokenize(texts).cuda()
        
                    class_embeddings = clip_model.encode_text(texts)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    if reduce=='mean':
                        class_embedding = class_embeddings.mean(dim=0)
                        class_embedding /= class_embedding.norm()
                        clip_weights.append(class_embedding)
                    if reduce is None:
                        class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                        clip_weights.append(class_embeddings)
                    i+=1
                indices.append(i)
                
            return clip_weights, indices
        else:
        
            for classname in classnames:
                
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                
                if gpt:
                    texts = template[classname]
                else:
                    texts = [t.format(classname)  for t in template]
                texts = clip.tokenize(texts).cuda()
                
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                if reduce=='mean':
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    clip_weights.append(class_embedding)
                if reduce is None:
                    class_embeddings /= class_embeddings.norm(dim=1, keepdim=True)
                    clip_weights.append(class_embeddings)
        
            clip_weights = torch.stack(clip_weights, dim=-1).cuda()
    return clip_weights


def get_samples_feature_and_labels(cache_dir, splits = ['test'], backbone_name = 'ViT-B/16', dataset_name = ''):
    model_cache_name = backbone_name.replace('/', '_')
    model_cache_name = model_cache_name.replace('-', '_')
    out = []
        
    for spl in splits:
        features_path = os.path.join(cache_dir, f'{model_cache_name}_{spl}_features.pt')
        try:
            _features = torch.load(features_path).cuda()
        except FileNotFoundError():
            raise FileNotFoundError(f'Could not find cached features at {features_path}. Run compute_features.py or check the --root_cache_path argument. ')
        _labels = torch.load(os.path.join(cache_dir, f'{spl}_target.pt')).cuda()
        out.append(_features)
        out.append(_labels)
    
    return out




def get_all_features(cfg, train_loader, val_loader, test_loader, dataset, clip_model):
    clip_prototypes = clip_classifier(dataset.classnames, dataset.template, clip_model, reduce=None)
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    
    if val_loader is not None:
        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    else:
        val_features, val_labels = None, None
        
    shot_features = None
    shot_labels = None

    if cfg['shots'] > 0:
        shot_features, shot_labels = build_cache_model(cfg, clip_model, train_loader, n_views=0,
                                                     reduce=None)
        #val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)
    return shot_features, shot_labels, val_features, val_labels, test_features, test_labels, clip_prototypes


def build_cache_model(cfg, clip_model, train_loader_cache, n_views=0, reduce=None):
    print('... for shot samples from train split:')

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []
        if n_views == 0:
            n_epochs =1
        else:
            n_epochs = n_views
        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(n_epochs):
                train_features = []
                train_labels = []
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                        
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))


        
        if n_views == 1:
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            #cache_keys = cache_keys.permute(1, 0)
        else:
            cache_keys = torch.cat(cache_keys, dim=0) # [n_views, n_classes, n_features]
            if reduce == 'mean':
                cache_keys = cache_keys.mean(0, keepdim=True)
                
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys.permute(0, 2, 1)
            
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values






def pre_load_features(cfg, split, clip_model, loader, n_views=1, mode = 'new'):

    #print('... from {} split:'.format(split))
    if cfg['load_pre_feat'] == False:
        if mode == 'new':
            batch_size = loader.batch_size
            num_samples = len(loader.dataset)
            features, labels = [], []
            
            with torch.no_grad():
              
                for view in range(n_views):
                    length = 0
                    i = 0
                    #for i, (images, target) in enumerate(tqdm(loader)):
                    pbar = tqdm(total = num_samples)
                    while i < num_samples:
                        j = min(i+batch_size, num_samples)
                        images = torch.vstack([loader.dataset[k][0][None,...] for k in range(i,j)])
                        target = torch.cat([torch.tensor(loader.dataset[k][1], dtype = torch.int64)[None] for k in range(i,j)])
                        if len(images.shape)==3:
                            images = images[None,...]
                        if n_views == 1:
                            
                            images, target = images.cuda(), target.cuda()
                            
                            
                            image_features = clip_model.encode_image(images)
                            
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            
                            
                            features.append(image_features.cpu())
                            labels.append(target.cpu())
                        else:
                            images, target = images.cuda(), target.cuda()
                            image_features = clip_model.encode_image(images)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            if view == 0:
                                labels.append(target.cpu())
                                if i ==0:
                                    mean_features = image_features
                                else:
                                    mean_features = torch.cat((mean_features, image_features))
                            else:
                                mean_features[length:length+image_features.size(0)] += image_features
                                length += image_features.size(0)
                        pbar.update(batch_size)
                        i = j
        else:
            features, labels = [], []
            with torch.no_grad():
                for view in range(n_views):
                    length = 0
                    for i, (images, target) in enumerate(tqdm(loader)):
                        if n_views == 1:
                            
                            images, target = images.cuda(), target.cuda()
                            
                            
                            image_features = clip_model.encode_image(images)
                            
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            
                            
                            features.append(image_features.cpu())
                            labels.append(target.cpu())
                        else:
                            images, target = images.cuda(), target.cuda()
                            image_features = clip_model.encode_image(images)
                            image_features /= image_features.norm(dim=-1, keepdim=True)
                            if view == 0:
                                labels.append(target.cpu())
                                if i ==0:
                                    mean_features = image_features
                                else:
                                    mean_features = torch.cat((mean_features, image_features))
                            else:
                                mean_features[length:length+image_features.size(0)] += image_features
                                length += image_features.size(0)
                                
        if n_views > 1:
            mean_features = mean_features / n_views
            features = mean_features / mean_features.norm(dim=-1, keepdim=True)
            labels = torch.cat(labels)
        
        elif n_views==1:
            features = torch.cat(features)
            labels = torch.cat(labels)
        
                
        backbone_name = cfg['backbone'].replace('/', '_')
        backbone_name = backbone_name.replace('\\', '_')
        backbone_name = backbone_name.replace('-', '_')
        print()
        torch.save(features, cfg['cache_dir'] + "/" +  f'{backbone_name}_{split}_features.pt')
        if not(os.path.exists(cfg['cache_dir'] + "/" +  f'{split}_target.pt')):
            torch.save(labels, cfg['cache_dir'] + "/" +  f'{split}_target.pt')
            
    else:
        print('LOADING FEATURES')
        try:
            features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
            labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
        except FileNotFoundError:
            backbone_name = cfg['backbone'].replace('/', '_')
            backbone_name = backbone_name.replace('\\', '_')
            backbone_name = backbone_name.replace('-', '_')
            features = torch.load(os.path.join(cfg['cache_dir'], f'{backbone_name}_{split}_features.pt'))
            labels = torch.load(os.path.join(cfg['cache_dir'], f'{split}_target.pt'))
    
    return features, labels


#%%

datasets = {
    'sun397':'SUN397',
    'imagenet':'imagenet',
    'fgvc_aircraft':'fgvc_aircraft',
    'eurosat':'eurosat',
    'food101':'Food101',
    'caltech101':'Caltech101',
    'oxford_pets':'OxfordPets',
    'oxford_flowers':'Flower102',
    'stanford_cars':'StanfordCars',
    'dtd':'DTD',
    'ucf101':'UCF101',
             }

backbones = {
             'vit_b16': 'ViT-B/16',
             'rn50': 'RN50',
             'vit_b32':'ViT-B/32',
             'rn101': 'RN101',                 
            }

#%%
def get_ensemble_prompt(cname):

    texts = [
        f'itap of a {cname}.',
        f'a bad photo of the {cname}.',
        f'a origami {cname}.',
        f'a photo of the large {cname}.',
        f'a {cname} in a video game.',
        f'art of the {cname}',
        f'a photo of the small {cname}'
        ]
    return texts

#%% 

def load_clip_classifier(dataset_name, root_prompts_path, 
                         prompts_types = 'custom', 
                         dataset = None, prompts_n_shots = 4, seed = 1, backbone = 'vit_b16', 
                         clip_model = None, num_classes = None, model_dim = 512,):
    if prompts_types == 'standard':
        clip_prototypes = clip_classifier(dataset.classnames, dataset.template, clip_model)
    elif prompts_types == 'coop-fewshot':
        path = os.path.join(root_prompts_path, 'Few_shot', 'coop', backbone, f'{prompts_n_shots}shots',
                            dataset_name, f'seed{seed}')
        clip_prototypes = torch.load(os.path.join(path, 'text_features.pt')).T
    elif prompts_types == 'taskres-fewshot':
        path = os.path.join(root_prompts_path, 'Few_shot', 'taskres', backbone, f'{prompts_n_shots}shots',
                            dataset_name, f'seed{seed}')
        clip_prototypes = torch.load(os.path.join(path, 'text_features.pt')).T
    elif prompts_types == 'coop-16shotsimagenet':
        path = os.path.join(root_prompts_path, 'Cross-dataset', 'coop', backbone, f'{prompts_n_shots}shots',
                            dataset_name, f'seed{seed}')
        clip_prototypes = torch.load(os.path.join(path, 'text_features.pt')).T
    elif prompts_types == 'cupl':
        # Load CuPL prompts
        prompt_datasetname = dataset_name.replace('_', '')
        if 'flowers' in dataset_name:
            prompt_datasetname = 'flowers102'
        prompt_path = os.path.join(root_prompts_path, 'cupl', f"CuPL_prompts_{prompt_datasetname}.json")
        with open(prompt_path, 'r') as f:
            cupl_prompts = json.load(f)
        
        # Encode cupl prompts
        tokenized_prompts = {}
        K = num_classes
        d = model_dim
        num_prompts = torch.zeros(K)
        avg_prompts = torch.zeros((K,d), dtype = torch.float16)
        
        for j,cname_ in enumerate(dataset.classnames):
            cname = cname_.replace('_', ' ')
            #print(f'class : {cname}, num_prompts : {len(cupl_prompts[cname])}')
            tokenized_prompts[cname] = clip.tokenize(cupl_prompts[cname]).cuda()
            num_prompts[j] = len(cupl_prompts[cname])
            with torch.autocast("cuda"), torch.no_grad():
                encoded_p_cname = clip_model.encode_text(tokenized_prompts[cname].cuda())
            encoded_p_cname = encoded_p_cname/torch.linalg.norm(encoded_p_cname, dim = -1, keepdims = True)
            avg_prompts[j,...] = encoded_p_cname.mean(0)
        clip_prototypes = (avg_prompts/torch.linalg.norm(avg_prompts, dim = -1, keepdims = True)).T.cuda()
    elif prompts_types == 'custom_ensemble':
        # Encode prompts
        tokenized_prompts = {}
        K = num_classes
        d = model_dim
        num_prompts = torch.zeros(K)
        avg_prompts = torch.zeros((K,d), dtype = torch.float16)
        
        for j,cname_ in enumerate(dataset.classnames):
            cname = cname_.replace('_', ' ')
            #print(f'class : {cname}, num_prompts : {len(cupl_prompts[cname])}')
            txts = get_ensemble_prompt(cname)
            tokenized_prompts[cname] = clip.tokenize(txts).cuda()
            num_prompts[j] = len(txts)
            with torch.autocast("cuda"), torch.no_grad():
                encoded_p_cname = clip_model.encode_text(tokenized_prompts[cname].cuda())
            encoded_p_cname = encoded_p_cname/torch.linalg.norm(encoded_p_cname, dim = -1, keepdims = True)
            avg_prompts[j,...] = encoded_p_cname.mean(0)
        clip_prototypes = (avg_prompts/torch.linalg.norm(avg_prompts, dim = -1, keepdims = True)).T.cuda()
    return clip_prototypes.type(torch.float16)

#%%
import datasets as dts
def load_features(dataset_name, 
                  root_path, 
                  cache_dir, 
                  preprocess, 
                  clip_model, 
                  backbone_name,
                  splits = ['train', 'test'],
                  load_loaders = True):
    cfg = {}
    print(f'============ DATASET : {dataset_name}')
    
    cfg['dataset'] = datasets[dataset_name]
    
    cfg['root_path'] = root_path 
    cfg['shots'] = 0
    cfg['load_pre_feat'] = True
    cfg['cache_dir'] = cache_dir
    if dataset_name == 'imagenet':
        cfg['load_cache'] = False
    if load_loaders:
        train_loader, val_loader, test_loader, dataset = dts.get_all_dataloaders(cfg, preprocess, dirichlet=None)
    else:
        dataset = dts.dataset_list[dataset_name](cfg['root_path'], cfg['shots'])
        train_loader, val_loader, test_loader = None,None,None
    features_and_labels = get_samples_feature_and_labels(cache_dir,
                                                            splits = splits,
                                                            backbone_name = backbone_name,
                                                            dataset_name = dataset_name)
    
        
    return train_loader, val_loader, test_loader, dataset, features_and_labels

#%%
def get_default_dmn_args():
    class Args:
        def __init__(self):
            return None
    dmn_args = Args()
    dmn_args.indice = 0
    dmn_args.shared_param = None
    dmn_args.mapping = 'bias'
    dmn_args.position = 'all'
    dmn_args.n_shot = 0 #zero shot
    dmn_args.n_augments = 0
    dmn_args.selection_p = 0.1
    return dmn_args