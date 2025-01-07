# -*- coding: utf-8 -*-
import argparse
import clip
import os
import utils as uti
import datasets as dts
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str)
    parser.add_argument('-d', '--datasets', default = ['dtd'], nargs="*", help = 'List of datasets for which to compute features.')
    parser.add_argument('--backbone', default='vit_b16', type=str, help = 'Name of the backbone to use. Examples : vit_b16 or rn101.')
    parser.add_argument('--root_cache_path', default = None, type = str, help = 'Path where the cached features and targets will be stored. Defaults to data_root_path/{dataset}/cache internally.')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    assert args.data_root_path is not None
    cfg = {}
    cfg['backbone'] = uti.backbones[args.backbone]
    print('========== Loading Clip Model')
    clip_model, preprocess = clip.load(cfg['backbone'])
    if args.root_cache_path is not None:
        base_cache_dir = args.root_cache_path
    else:
        base_cache_dir = args.data_root_path
    for dataset_name in args.datasets:
        print('\n******* dataset : ', dataset_name)
        if dataset_name == 'imagenet':
            cfg['load_cache'] = True
        cfg['dataset'] = uti.datasets[dataset_name]
        cfg['root_path'] = args.data_root_path
        cfg['shots'] = 0
        cfg['load_pre_feat'] = False
        cache_dir = os.path.join(base_cache_dir, uti.datasets[dataset_name], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cfg['cache_dir'] = cache_dir
        print(cfg['cache_dir'])
        print('Computing Features...')
        train_loader, val_loader, test_loader, dataset = dts.get_all_dataloaders(cfg, preprocess, dirichlet=None)
        _ = uti.get_all_features(
            cfg, train_loader, val_loader, test_loader, dataset, clip_model)
    return None

if __name__ == '__main__':
    main()
