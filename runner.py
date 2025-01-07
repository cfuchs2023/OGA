

def run(args):
    print('Beginning import')
    import torch
    import os
    from tqdm import tqdm
    import clip
    import utils as uti
    # from utils import datasets #Dict with all possible datasets
    # from utils import backbones #Dict with the possible backbones
    import pickle  
    import json    
    from OGA import OGA_core 
    from TDA import TDA_core as tdac
    import numpy as np
    import DMN.DMN_clip_wrapper as DMN_clip_wrapper
    import DMN.DMN_core as DMN_core
    print('Import done')
    
    
    # ==================== Get arguments
    root_path = args.data_root_path
    
    if args.root_save_path is not None:
        base_save_path = args.root_save_path
    else:
        base_save_path = os.path.join(root_path, 'results')
        os.makedirs(base_save_path, exist_ok = True)
    
    if args.root_cache_path is not None:
        base_cache_dir = args.root_cache_path
    else:
        base_cache_dir = root_path
    
    if args.root_prompts_path is not None:
        root_prompts_path = args.root_prompts_path
    else:
        root_prompts_path = root_path
        
    model_name = args.backbone
    n_runs = args.n_runs #100  #only for random tasks
    
    prompts_types = args.prompts_types
    prompts_n_shots = args.prompts_n_shots
    prompts_seed = args.prompts_seed
    batch_size = args.batch_size
    shot_capacity = args.shot_capacity # size of the memory in shots per class
    
    check_on_fullset = args.check_on_fullset #Evaluate accuracy of the adapted model on the complete test set every X batches
    adapt_method_name = args.adapt_method_name 
    
    # ==================== Generate tasks seeds from master seed
    rng = np.random.default_rng(args.master_seed)
    shuffles_seed = rng.choice(range(10000*n_runs), size = n_runs, replace = False)
    
    
    # ==================== Load clip model
    clip_model, preprocess = clip.load(uti.backbones[model_name])
    
    # =================== Get run name for saving results
    if args.run_name is not None:
        run_name = args.run_name
    else:
        if prompts_types in ['coop-fewshot', 'taskres-fewshot']:
            pname = prompts_types.split('-')[0]
            prompts_suffix = f'prompts_{pname}_seed_{prompts_seed}_shots_{prompts_n_shots}'
        else:
            pname = prompts_types
            prompts_suffix = f'prompts_{pname}'
            
        if adapt_method_name == 'OGA':
            method_suffix = f'OGA_sig_{args.OGA_params.sig_type}_shot_capacity_{shot_capacity}'
        elif adapt_method_name == 'TDA':
            method_suffix = f'TDA_shot_capacity_{shot_capacity}_neg_capacity_{args.TDA_params.neg_cache_capacity}'
        elif adapt_method_name == 'DMN':
            method_suffix = f'DMN_shot_capacity_{shot_capacity}'
            
            
        run_name = f'{method_suffix}_{prompts_suffix}'
    
    # ==================== Prepare results dict with some arguments
    resu = {} #Dict storing all results
    if adapt_method_name == 'OGA':
        tau = args.OGA_params.tau
        if type(tau) is float:
            resu['tau'] = tau #.cpu().numpy().tolist()
        else:
            resu['tau'] = tau.cpu().numpy().tolist()
            
    # ==================== Loop over requested datasets
    for dataset_name in args.datasets:
        resu[dataset_name] = {}
        resu[dataset_name][adapt_method_name] = {}
        if check_on_fullset:
            if dataset_name == 'food101':
                check_on_fullsets_interval = 15
            elif dataset_name == 'stanford_cars':
                check_on_fullsets_interval = 10
            else:
                check_on_fullsets_interval = 5
        
        # ******* Load features and labels
        if base_cache_dir is None:
            cache_dir = os.path.join(root_path, uti.datasets[dataset_name], 'cache')
        else:
            cache_dir = os.path.join(base_cache_dir, uti.datasets[dataset_name], 'cache')
            
        train_loader, val_loader, test_loader, dataset,\
        features_and_labels\
        = uti.load_features(dataset_name, 
                            root_path, 
                            cache_dir, 
                            preprocess, 
                            clip_model,
                            uti.backbones[model_name],
                            splits = ['test'])
        test_features, test_labels = features_and_labels
        K = torch.max(test_labels)+1
        d = test_features.shape[-1]
        
        # ******* Load encoded textual prompts
        clip_prototypes = uti.load_clip_classifier(dataset_name, 
                                                   root_prompts_path, 
                                 prompts_types = prompts_types,
                                 dataset = dataset, 
                                 prompts_n_shots = prompts_n_shots, 
                                 seed = prompts_seed, 
                                 backbone = model_name, 
                                 num_classes = K,
                                 model_dim = d,
                                 clip_model = clip_model, )

        
        # ******* Check zero-shot performance
        temp = 100
        zs_logits = torch.zeros((test_features.shape[0], K), dtype = torch.float64, device = test_features.device)
        zs_logits[...] = temp * test_features@clip_prototypes.squeeze()
        zs_pred = torch.argmax(zs_logits, dim = -1).cuda()
        zs_acc = torch.sum(zs_pred.cpu() == test_labels.cpu())/zs_pred.shape[0]
        print(f'Zero-shot accuracy : {zs_acc}')
        resu[dataset_name]['zero_shot'] = zs_acc.item() 
        
        # ******* Cache zero-shot soft labels and zero shot entropies
        zs_probs = zs_logits.softmax(-1).cuda()
        zs_entropy = - torch.sum(torch.log(zs_probs+1e-9)*zs_probs, dim = -1)
        
        
        # ******* Prepare list of results
        all_acc_batchs_all_tasks_ = []
        if check_on_fullset: # If evaluating on full test set at regular intervals is requested
            all_full_datasets_accuracies_ = []
        
        # ******* Loop over tasks
        for n_ in tqdm(range(shuffles_seed.shape[0])):
            if check_on_fullset:
                all_full_datasets_accuracies_.append([])
            rng = np.random.default_rng(seed = shuffles_seed[n_])
            shuffle = torch.tensor(rng.choice(range(test_features.shape[0]), 
                                               test_features.shape[0], 
                                               replace = False))
            
            # ******* Initialize online adaptation model
            if adapt_method_name == 'OGA':
                oga_model = OGA_core.GaussAdapt(clip_prototypes.squeeze().T, 
                                             shot_capacity = shot_capacity, 
                                             sig_type = args.OGA_params.sig_type).cuda()
            elif adapt_method_name == 'TDA':
                TDA_neg_cache = tdac.TDA_NegCache(K, d, shot_capacity=args.TDA_params.neg_cache_capacity).cuda()
                TDA_pos_cache = tdac.TDA_PosCache(K, d, shot_capacity = shot_capacity - args.TDA_params.neg_cache_capacity).cuda()
                TDA_pos_cache_logits_scale = args.TDA_params.pos_cache_logits_scale
                TDA_neg_cache_logits_scale = args.TDA_params.neg_cache_logits_scale
                
            elif adapt_method_name == 'DMN':
                dmn_args = uti.get_default_dmn_args()
                dmn_args.memory_size = shot_capacity
                DMN_clip = DMN_clip_wrapper.DMNClipWrapper(clip_model, 
                          preprocess, 
                          'cuda', 
                          dataset.classnames, 
                          batch_size, 
                          arch = uti.backbones[model_name],
                          memory_size = shot_capacity)
                DMN_clip.reset_classnames(dataset)
                dmn = DMN_core.DualMem(dmn_args, feat_dim = test_features.shape[-1], class_num = K)
                dmn = dmn.cuda()
                DMN_clip.eval()
                dmn.eval()
                with torch.autocast("cuda", dtype = torch.float16), torch.no_grad():
                    text_feat, text_feat_full = DMN_clip.get_text_features()
            # ******* Initialize task
            start = 0 # Start of current batch slice
            end = 0 # End of current batch slice
            num_batch = 1 # Current batch number
            all_acc_batches_single_task_ = [] # Per batch accuracy of current task
            last_batch = False
            
            # ******* Loop over batches
            while not(last_batch):
                
                # ******* Get indexes of current batch
                start = end
                end = min(start+batch_size, test_features.shape[0])
                indices = shuffle[start:end]
                if end == test_features.shape[0]:
                    last_batch = True
                    
                # ******* Evaluate accuracy on full test set if need be
                if check_on_fullset and (not(num_batch % check_on_fullsets_interval) or last_batch):
                    if adapt_method_name == 'OGA':
                        log_probs = oga_model.get_log_probs(test_features)
                        z, p = oga_model.get_MAP(zs_probs, log_probs, tau = tau, simplex_p = True)
                        pred = torch.argmax(z, dim = 1)
                    elif adapt_method_name == 'TDA':
                        pos_cache_logits = TDA_pos_cache.get_logits(test_features, beta = 5, alpha = 1) 
                        neg_cache_logits = TDA_neg_cache.get_logits(test_features, beta = 1) 
                        query_logits = zs_logits + TDA_pos_cache_logits_scale * pos_cache_logits + TDA_neg_cache_logits_scale * neg_cache_logits
                        pred = torch.argmax(query_logits, dim = -1)
                        
                    acc = torch.sum(pred == test_labels, dim = 0)/test_labels.shape[0]
                    all_full_datasets_accuracies_[-1].append([num_batch-1, acc.cpu().item()])
                    
                # ******* Get batch data
                batch_features = test_features[indices, :]
                batch_labels = test_labels[indices]
                batch_zs_pseudo_labels = zs_pred[indices]
                batch_zs_probs = zs_probs[indices]
                batch_zs_logits = zs_logits[indices, :]
                batch_zs_entropy = zs_entropy[indices]
                 
                # ******* Update memory on batch
                if adapt_method_name == 'OGA':
                    _ = oga_model.update_memory(batch_features,
                                                                            batch_zs_logits, 
                                                                            batch_zs_probs,
                                                                            batch_zs_entropy,
                                                                            batch_zs_pseudo_labels,
                                                                            tau = args.OGA_params.tau,
                                                                            normalize_mu = args.OGA_params.normalize_mu) 
                elif adapt_method_name == 'TDA':
                    _ = TDA_pos_cache.update_memory(batch_features, batch_zs_logits)
                    _ = TDA_neg_cache.update_memory(batch_features, batch_zs_logits)
                elif adapt_method_name == 'DMN':
                    with torch.autocast("cuda"), torch.no_grad():
                       for ju,u in enumerate(indices):
                               DMN_clip.image_features_global = batch_features[ju:ju+1,...]
                               # We never use augmentations
                               # confidence_prediction, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text.softmax(1),
                               #                                                                             dmn_args.selection_p)
                               dmn.init_pred = batch_zs_probs[ju:ju+1,:]
                               dmn.update_memory_bank(DMN_clip) 

                # ******* Predict on batch using updated memory
                if adapt_method_name == 'OGA':
                    log_probs = oga_model.get_log_probs(batch_features)
                    z, p = oga_model.get_MAP(batch_zs_probs, log_probs, tau = args.OGA_params.tau, simplex_p = True)
                    pred = torch.argmax(z, dim = 1)
                elif adapt_method_name == 'TDA':
                    pos_cache_logits = TDA_pos_cache.get_logits(batch_features, beta = 5, alpha = 1) 
                    neg_cache_logits = TDA_neg_cache.get_logits(batch_features, beta = 1) 
                    query_logits = batch_zs_logits + TDA_pos_cache_logits_scale * pos_cache_logits + TDA_neg_cache_logits_scale * neg_cache_logits
                    pred = torch.argmax(query_logits, dim = -1)
                elif adapt_method_name == 'DMN':
                    with torch.autocast("cuda"), torch.no_grad(): #
                        all_img_logits = dmn.fast_get_image_pred(batch_features, DMN_clip, clip_prototypes)
                        all_img_probs = all_img_logits.softmax(-1)
                    final_probs = batch_zs_probs + args.DMN_params.DMN_prob_factor * all_img_probs      
                    pred = torch.argmax(final_probs, dim = -1)
                    acc = torch.sum(pred == batch_labels, dim = 0)/batch_labels.shape[0]
                    all_acc_batches_single_task_.append(acc.cpu())
                acc = torch.sum(pred == batch_labels, dim = 0)/batch_labels.shape[0]
                
                # ******* Store accuracy on current batch
                all_acc_batches_single_task_.append(acc.cpu())
                num_batch += 1
                
                
            all_acc_batches_single_task = torch.stack(all_acc_batches_single_task_)
            all_acc_batchs_all_tasks_.append(all_acc_batches_single_task)
            print(f'{adapt_method_name} average accuracy on task number {n_} : ', all_acc_batches_single_task.mean(dim=0).cpu())
            
        # ******* Prepare results for disk writing
        all_acc_batchs_all_tasks = torch.stack(all_acc_batchs_all_tasks_)
        avg_per_task_accs = torch.mean(all_acc_batchs_all_tasks, dim = 1)
        avg_accs = torch.mean(avg_per_task_accs, dim = 0)
        resu[dataset_name][adapt_method_name]['avg_acc'] = avg_accs.cpu().numpy().tolist()
        if check_on_fullset:
            resu[dataset_name][adapt_method_name]['checks_on_full_dataset'] = torch.tensor(all_full_datasets_accuracies_)
       
        
        ignore_keys = ['all_acc_batchs_all_tasks', 
               'checks_on_full_dataset']
        partial_resu = {} # Resu with only avg acc per dataset
        for dname in resu.keys():
            if type(resu[dname]) is dict:
                partial_resu[dname] = {}
                partial_resu[dname]['zero_shot'] = resu[dname]['zero_shot']
                for method_key in resu[dname].keys():
                    if type(resu[dname][method_key]) is dict:
                        partial_resu[dname][method_key] = {}
                        for key in resu[dname][method_key].keys():
                            if key not in ignore_keys:
                                partial_resu[dname][method_key][key] = resu[dname][method_key][key]
                    else:
                        partial_resu[dname][method_key] = resu[dname][method_key]
            else:
                partial_resu[dname] = resu[dname]
                
        # ******* Store average acc / dataset in a json
        with open(os.path.join(base_save_path, run_name+'.json'), 'w') as f:
            json.dump(partial_resu,f)
            
        # ******* Store complete results in a pickle
        with open(os.path.join(base_save_path, run_name+'.pickle'), 'wb') as f:
            pickle.dump(resu,f)

    return None