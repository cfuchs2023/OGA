# -*- coding: utf-8 -*-
import torch 
class GaussAdapt(torch.nn.Module):
    def __init__(self, clip_prototypes, shot_capacity = 8, sig_type = 'RidgeMoorePenrose'):
        '''shot_capacity: maximum number of stored samples per class.
        sig_type: type of estimator for teh covariance. One of 'Ridge', 'MoorePenrose' or the recommended 'RidgeMoorePenrose'. 
        The latter transitions from empirical Bayes Ridge (see https://doi.org/10.1016/j.jmva.2008.01.016) to inverse when more than 4d sampels are available. '''
        super(GaussAdapt, self).__init__()
        assert sig_type in ['RidgeMoorePenrose', 'Ridge', 'MoorePenrose']
        K,d = clip_prototypes.shape
        self.shot_capacity = shot_capacity
        self.K = K
        self.clip_prototypes = clip_prototypes #should be (K,d)
        self.mus = clip_prototypes.clone().type(torch.float32)
        self.temp = 100
        self.d = clip_prototypes.shape[-1]
        self.count = torch.nn.Parameter(torch.zeros(self.K), requires_grad = False)
        self.sig_type = sig_type
        self.Sig = torch.nn.Parameter(1/d * torch.eye(d, dtype = torch.float32), requires_grad = False)
        self.inv_Sig = torch.nn.Parameter(d * torch.eye(d, dtype = torch.float32), requires_grad = False)
        self.memory_state = torch.nn.Parameter(torch.zeros((K,shot_capacity), dtype = torch.bool), requires_grad = False)
        
        self.memory = torch.nn.Parameter(torch.zeros((K, shot_capacity, d), dtype = torch.float16), 
                                         requires_grad = False)
        self.memory_soft_labels = torch.nn.Parameter(torch.zeros((K, shot_capacity), dtype = torch.float16), 
                                         requires_grad = False)
        self.__init_entropy(prop_max = 1)

        return None
    
    def __init_entropy(self, prop_max = 1):
        max_entropy = -torch.log(torch.tensor(1/self.K))
        init_val = prop_max * max_entropy
        self.memory_entropy = torch.nn.Parameter(init_val * torch.ones((self.K,self.shot_capacity), dtype = torch.float16, device = self.memory.device),
                                                 requires_grad = False)
        return init_val
    
        
    def get_entropy(self, probs):
        sh_entropy = - torch.sum(torch.log(probs+1e-6)*probs, dim = -1)
        return sh_entropy
    
    def __update_memory_entropy(self, x, text_prob, entropy, pseudo_label, gauss_prob = None):
        updated = False
        if torch.any(entropy<self.memory_entropy[pseudo_label,:]):
            idx_max = torch.argmax(self.memory_entropy[pseudo_label,:])
            #print(f'Replaced memorized sample {pseudo_label} for class {pseudo_label}')
            self.memory[pseudo_label, idx_max] = x[...]
            self.memory_entropy[pseudo_label, idx_max] = entropy
            self.memory_state[pseudo_label, idx_max] = True
            self.memory_soft_labels[pseudo_label,idx_max] = text_prob[pseudo_label]
            updated = True
    
        return updated
    

            
    def update_memory(self,
                      features,
                      text_logits, 
                      zs_probs, 
                      zs_entropy,
                      zs_labels,
                      tau = 0.03,
                      normalize_mu = False):
        '''This method updates the memory as well as the means and covariance if necessary. '''
        selected_samples = []
        updated = False
        
        # update labels
        for ji in range(zs_labels.shape[0]):
            up = self.__update_memory_entropy(features[ji,:], zs_probs[ji,:], zs_entropy[ji], zs_labels[ji])
            if up:
                selected_samples.append(ji)
                updated = True

        if updated:
            self.__update_mu(normalize_mu = normalize_mu)
            self.__update_sigma()
            
        return updated, selected_samples #, zs_entropy, upd_entropy
    
    def __update_mu(self, normalize_mu = False):
        #means = torch.mean(self.memory_state[...,None]*self.memory, dim = 1).float()
        
        means = torch.mean(self.memory_state[...,None]*self.memory, dim = 1).float()
        mask = torch.sum(self.memory_state,dim=1)>=2 # was >2
        self.mus[mask,:] = means[mask,:].type(torch.float32)
        if normalize_mu:
            self.mus[mask,:] = self.mus[mask,:] / torch.linalg.norm(self.mus[mask,:], dim = -1, keepdims = True)
        return None
    
    
    def __update_sigma(self, use_soft_labels = False):      
        if 'Ridge' == self.sig_type:
            d = self.mus.shape[-1]
            x = self.memory.view((self.K*self.shot_capacity, d))
            x_mem_state = self.memory_state.view((self.K*self.shot_capacity))
            if torch.any(torch.sum(self.memory_state, dim = -1)>2):
                x_labels = torch.tensor([k for k in range(self.K) for _ in range(self.shot_capacity)], device = x.device)
                center_vecs = torch.cat([x[torch.logical_and(x_mem_state, x_labels == k)] - self.mus[k:k+1,:] for k in range(self.K)])
                M = center_vecs.T.cov()
                trace = torch.sum(M[range(d), range(d)])
                # shape 1 = d / shape 0 = n
                n,d = center_vecs.shape
                cov_inv = d * torch.linalg.pinv((n - 1) * M + trace * torch.eye(d, device = center_vecs.device))    
                self.Sig[...] = M
                self.inv_Sig[...] = cov_inv
        elif 'RidgeMoorePenrose' == self.sig_type:
            d = self.mus.shape[-1]
            n = torch.sum(self.memory_state)
            
            if torch.any(torch.sum(self.memory_state, dim = -1)>2):
                x = self.memory.view((self.K*self.shot_capacity, d))
                x_labels = torch.tensor([k for k in range(self.K) for _ in range(self.shot_capacity)], device = x.device)
                x_mem_state = self.memory_state.view((self.K*self.shot_capacity))
                
                class_probs = self.memory_soft_labels[self.memory_state]
                center_vecs = torch.cat([x[torch.logical_and(x_mem_state, x_labels == k)] - self.mus[k:k+1,:] for k in range(self.K)])
                center_vec_mean = center_vecs.mean(dim=0)
                if use_soft_labels:
                    #M = center_vecs.T.cov(correction=1)
                    c_center_vecs = (center_vecs - center_vec_mean[None,:]) * class_probs[:,None]
                    M = c_center_vecs.T @ c_center_vecs / torch.sum(class_probs)
                else:
                    c_center_vecs = (center_vecs - center_vec_mean[None,:])
                    M = c_center_vecs.T @ c_center_vecs / (n-1)
                
                if n<=4*d:
                    # use shrinkage
                    
                    trace = torch.sum(M[range(d), range(d)])
                    # shape 1 = d / shape 0 = n
                    cov_inv = d * torch.linalg.pinv((n - 1) * M + trace * torch.eye(d, device = center_vecs.device))    
                    self.Sig[...] = M
                    self.inv_Sig[...] = cov_inv
                else:
                    # Use pinv
                    self.Sig[...] = M
                    self.inv_Sig[...] = torch.linalg.pinv(M.type(torch.float32))
        elif 'MoorePenrose' == self.sig_type:
            d = self.mus.shape[-1]
            x = self.memory.view((self.K*self.shot_capacity, d))
            x_mem_state = self.memory_state.view((self.K*self.shot_capacity))
            if torch.any(torch.sum(self.memory_state, dim = -1)>2):
                x_labels = torch.tensor([k for k in range(self.K) for _ in range(self.shot_capacity)], device = x.device)
                center_vecs = torch.cat([x[torch.logical_and(x_mem_state, x_labels == k)] - self.mus[k:k+1,:] for k in range(self.K)])
                M = center_vecs.T.cov()
                self.Sig[...] = M
                self.inv_Sig[...] = torch.linalg.pinv(M.type(torch.float32)) 
                
        return None
    
    def get_log_probs(self,x):
        W = torch.einsum('nd, dc -> cn', self.mus, self.inv_Sig)
        b =  - torch.einsum('nd, dc, nc -> n', self.mus, self.inv_Sig, self.mus) / 2
        Q =  - torch.einsum('nd, dc, nc -> n', x.float(), self.inv_Sig, x.float()) / 2
        log_probs = (x.float() @ W + b)
        log_probs += Q[:,None]
        return log_probs
        
    
    def get_MAP(self, y_hat, memory_logits, tau = 0.01, simplex_p = False):
        '''y_hat: zero shot soft labels. memory_logits: log probabilities obtained from the cached samples. '''
        lambd = 1.0
        assert type(tau) is float or type(lambd) is float
        # Compute gaussian probs
        if type(tau) is float:
            if not simplex_p:
                p_ = torch.exp(tau * memory_logits)
            else:
                p_ = (tau*memory_logits).softmax(-1)
        else:
            if not simplex_p:
                p_ = torch.exp(tau[None,None,:] * memory_logits[...,None])
            else:
                p_ = (tau[None,None,:] * memory_logits[...,None]).softmax(-1)
            
        # Compute MAP (only if y_hat is not None)       
        if y_hat is None:
            z = None
        else:
            if type(lambd) is float:
                if len(p_.shape) == 2:
                    z = (y_hat**lambd) * p_
                    z = z/torch.sum(z, dim = 1, keepdims = True)
                elif len(p_.shape) == 3:
                    z = (y_hat**lambd)[...,None] * p_
                    z = z/torch.sum(z, dim = 1, keepdims = True)
                else:
                    raise RuntimeError(f'Incompatible p_ shape {p_.shape}')
                    
            else:
                if len(p_.shape) == 2:
                    z = (y_hat[:,:,None]**lambd[None,None,:]) * p_[:,:,None]
                elif len(p_.shape) == 3:
                    z = (y_hat[:,:,None]**lambd[None,None,:])[...,None] * p_[...,None]
                else:
                    raise RuntimeError(f'Incompatible p_ shape {p_.shape}')
        return z, p_