# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import operator
from .TDA_utils import *

#%%
class TDA_PosCache(torch.nn.Module):
    def __init__(self, K, d, shot_capacity = 8):
        super(TDA_PosCache, self).__init__()
        self.shot_capacity = shot_capacity
        self.memory = torch.nn.Parameter(torch.zeros((K,shot_capacity,d), dtype = torch.float16),
                                      requires_grad = False)
        self.memory_entropy = torch.nn.Parameter(1e3 * torch.ones((K,shot_capacity), dtype = torch.float16),
                                      requires_grad = False)
        self.memory_state = torch.nn.Parameter(torch.zeros((K,shot_capacity), dtype = torch.bool), requires_grad = False)
        self.K = K
        self.d = d
        self.shot_capacity = shot_capacity
        # self.memory_indexes = torch.nn.Parameter(torch.zeros((K, shot_capacity), dtype = torch.int32),
        #                               requires_grad = False) # for sanity checks
        self.init_entropy(prop_max = 1)
        return None
    

    def init_entropy(self, prop_max = 1):
        max_entropy = -torch.log(torch.tensor(1/self.K).to(self.memory_entropy.device))
        init_val = prop_max * max_entropy
        self.memory_entropy = torch.nn.Parameter(init_val * torch.ones((self.K,self.shot_capacity), dtype = torch.float16, device = self.memory.device),
                                                 requires_grad = False)
        return init_val 
    def __update_memory(self, x, text_logit):
        text_prob = text_logit.softmax(-1)
        entropy = self.get_entropy(text_prob).item()
        text_label = torch.argmax(text_logit, dim = -1)
        updated = False
        if torch.any(entropy<self.memory_entropy[text_label,:]):
            idx_max = torch.argmax(self.memory_entropy[text_label,:])
            #print(f'Replaced memorized sample {text_label} for class {text_label}')
            self.memory[text_label, idx_max] = x[...]
            self.memory_entropy[text_label, idx_max] = entropy
            self.memory_state[text_label, idx_max] = True
            updated = True
        return updated
    
    def update_memory(self, x, text_logits, ):
        selected_samples = []
        for i in range(text_logits.shape[0]):
            up = self.__update_memory(x[i,:], text_logits[i,:])
            if up:
                selected_samples.append(i)
        return selected_samples
    
    def get_logits(self,x, beta = 5, alpha = 2):
        A_ = (self.memory @ x.T).permute((2,0,1))
        A = torch.exp(-beta * (1-A_))
        logits = torch.sum(A, dim = -1)
        return alpha * logits
    
    def get_entropy(self, probs):
        sh_entropy = - torch.sum(torch.log(probs+1e-6)*probs, dim = -1)
        return sh_entropy
    
    
class TDA_NegCache(torch.nn.Module):
    def __init__(self, K, d, shot_capacity = 8, 
                 upper_entropy_bound = 0.5, lower_entropy_bound = 0.2,
                 lower_probability_bound = 0.03):
        super(TDA_NegCache, self).__init__()
        self.shot_capacity = shot_capacity
        self.memory = torch.nn.Parameter(torch.zeros((K,shot_capacity,d), dtype = torch.float16),
                                      requires_grad = False)
        self.memory_entropy = torch.nn.Parameter(1e3 * torch.ones((K,shot_capacity), dtype = torch.float16),
                                      requires_grad = False)
        self.memory_state = torch.nn.Parameter(torch.zeros((K,shot_capacity), dtype = torch.bool), requires_grad = False)
        self.K = K
        self.d = d
        self.shot_capacity = shot_capacity
        self.lower_probability_bound = lower_probability_bound
        self.lower_entropy_bound = lower_entropy_bound
        self.upper_entropy_bound = upper_entropy_bound
        # self.memory_indexes = torch.nn.Parameter(torch.zeros((K, shot_capacity), dtype = torch.int32),
        #                               requires_grad = False) # for sanity check
        self.init_entropy(prop_max = 1)
        return None
    
    def init_entropy(self, prop_max = 1):
        max_entropy = -torch.log(torch.tensor(1/self.K).to(self.memory_entropy.device))
        init_val = prop_max * max_entropy
        self.memory_entropy = torch.nn.Parameter(init_val * torch.ones((self.K,self.shot_capacity), dtype = torch.float16, device = self.memory.device),
                                                 requires_grad = False)
        return init_val 
    
    def __update_memory(self, x, text_logit,):
        text_prob = text_logit.softmax(-1)
        text_label = torch.argmax(text_logit, dim = -1)
        updated = False
        if text_prob[text_label] > self.lower_probability_bound:
            entropy = self.get_entropy(text_prob).item()
            if entropy > self.lower_entropy_bound and entropy < self.upper_entropy_bound:
                if torch.any(entropy<self.memory_entropy[text_label,:]):
                    idx_max = torch.argmax(self.memory_entropy[text_label,:])
                    #print(f'Replaced memorized sample {text_label} for class {text_label}')
                    self.memory[text_label, idx_max] = x[...]
                    self.memory_entropy[text_label, idx_max] = entropy
                    self.memory_state[text_label, idx_max] = True
                    updated = True
        return updated
    
    def update_memory(self, x, text_logits, ):
        selected_samples = []
        for i in range(text_logits.shape[0]):
            up = self.__update_memory(x[i,:], text_logits[i,:])
            if up:
                selected_samples.append(i)
        return selected_samples
    
    def get_logits(self,x, beta = 1): #alpha = 0.117
        A_ = (self.memory @ x.T).permute((2,0,1))
        A = torch.exp(-beta * (1-A_))
        logits = torch.sum(-A, dim = -1)
        return logits
    
    def get_entropy(self, probs):
        sh_entropy = - torch.sum(torch.log(probs+1e-6)*probs, dim = -1)
        return sh_entropy
    
    
            
#%%
def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]

def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = [] # Cached features
        cache_values = [] # Classes present in the cache
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

def compute_tda_logits(pos_cfg, neg_cfg, query_features, clip_weights,
                       pos_cache = {}, neg_cache = {}, do_update_cache = False):
    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
    with torch.no_grad():
        clip_logits, loss, prob_map, pred = get_clip_logits(query_features, clip_weights)
        prop_entropy = get_entropy(loss, clip_weights)
        if do_update_cache:
            if pos_enabled:
                update_cache(pos_cache, pred, [query_features, loss], pos_params['shot_capacity'])
    
            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [query_features, loss, prob_map], neg_params['shot_capacity'], True)

        final_logits = clip_logits.clone()
        if pos_enabled and pos_cache:
            final_logits += compute_cache_logits(query_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(query_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))
    return final_logits, pos_cache, neg_cache


def run_test_tda(pos_cfg, neg_cfg, query_features, query_labels, clip_weights,
                 pos_cache = {}, neg_cache = {}):

    indices = [[i] for i in range(query_features.shape[0])]
    with torch.no_grad():
        accuracies = []

        #Test-time adaptation
        for i in range(query_features.shape[0]):
            indexes = indices[i]
            images_features = query_features[indexes]
            targets = query_labels[indexes]
            final_logits, pos_cache, neg_cache = compute_tda_logits(pos_cfg, neg_cfg, 
                                                                    images_features, clip_weights,
                                   pos_cache, neg_cache, do_update_cache = True)
            acc = cls_acc(final_logits, targets)  
            accuracies.append(acc)

        #print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        avg_accuracy = sum(accuracies)/len(accuracies)
        return avg_accuracy, pos_cache, neg_cache
