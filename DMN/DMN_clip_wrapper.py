# -*- coding: utf-8 -*-
import torch.nn as nn
from clip import load, tokenize
import torch
class DMNClipWrapper(nn.Module):
    def __init__(self, clip_model, transform, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        learned_cls=False, memory_size=10, text_prompt_type='custom'):
        super(DMNClipWrapper, self).__init__()
        self.clip = clip_model
        self.classnames = [name.replace("_", " ") for name in classnames]
        self.first_flag = True
        self.memory_size = memory_size
        self.return_local_feat = False
        if text_prompt_type != 'custom':
            raise RuntimeError('Only custom prompts are supported.')
        self.text_prompt_type = text_prompt_type

        self.logit_scale = self.clip.logit_scale.data
        self.text_feat = None
        self.few_shot_mem = False
        # self.n_cls = len(classnames)  ## 200
        # self.image_encoder = clip.visual
        # # ipdb.set_trace()
        # self.text_encoder = TextEncoder(clip)
        # # prompt tuning
        # self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        # self.criterion = criterion

        
    # @property
    # def dtype(self):
    #     return self.image_encoder.conv1.weight.dtype

    # # restore the initial state of the prompt_learner (tunable prompt)
    # def reset(self):
    #     self.prompt_learner.reset()
    #
    def reset_classnames(self, dataset):
        self.n_cls = len(dataset.classnames)  ## 200
        self.classnames = [name.replace("_", " ") for name in dataset.classnames]
        self.text_prompt = dataset.template
        # ipdb.set_trace()
        # name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [self.prompt_prefix + " " + name + "." for name in classnames] ## 200
        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)  ## torch.Size([200, 77])
        #
        # clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
        #
        # with torch.no_grad():
        #     embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)  ## torch.Size([200, 77, 512])
        #
        # self.token_prefix = embedding[:, :1, :] ## 200*1*512 前缀
        # self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS ## torch.Size([200, 72, 512]) 后缀
        #
        # self.name_lens = name_lens
        # self.tokenized_prompts = tokenized_prompts  ## torch.Size([200, 77])
        # self.classnames = classnames
        self.first_flag = True

    def get_text_features(self):
        ## get the text feature only once, multiple class & multiple prompt
        text_feat = []
        text_label = []
        count = 0
        for name in self.classnames:
            text_prompts = [template.format(name) for template in self.text_prompt]  # format with class
            if self.text_prompt_type =='tip_cupl':
                text_prompts += self.cupl_prompts[name]
            texts = tokenize(text_prompts).cuda()  # tokenize
            class_embeddings = self.clip.encode_text(texts)  # embed with text encoder
            class_embeddings_full = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding_mean = class_embeddings_full.mean(dim=0)
            class_embedding_mean /= class_embedding_mean.norm()
            text_feat.append(class_embedding_mean) ### 1024
            one_hot_target = torch.zeros(self.n_cls).to(class_embedding_mean.device)
            one_hot_target[count] = 1
            text_label.append(one_hot_target)  ## 1 * d, turn it to one hot labels.
            count = count + 1
        self.text_feat = torch.stack(text_feat, dim=0).cuda() ## N*1024
        self.text_label = torch.stack(text_label, dim=0).cuda()  ## N*N

        self.text_feat_full = self.text_feat ## not used.
        ######## 直接从这里找出 important text feat following APE. TO DO
        self.fixed_global_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_global_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C
        self.fixed_local_feat_vanilla = self.text_feat.clone().unsqueeze(1) ## N*1*C

        self.fixed_global_label = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label = self.text_label.clone().unsqueeze(1)
        self.fixed_global_label_vanilla = self.text_label.clone().unsqueeze(1)
        self.fixed_local_label_vanilla = self.text_label.clone().unsqueeze(1)

        if self.first_flag:  ## initlize
            self.image_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device)       ## 如果满了，把entropy 最高的扔出去
            self.image_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.image_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.image_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)

            self.local_feature_memory = torch.zeros(self.n_cls, self.memory_size, self.text_feat.shape[1]).to(self.text_feat.device)
            self.local_prediction_mem = torch.zeros(self.n_cls, self.memory_size, self.n_cls).to(self.text_feat.device)  ## category prediction.
            self.local_entropy_mem = torch.zeros(self.n_cls, self.memory_size).to(self.text_feat.device)   ## category prediction.
            self.local_feature_count = torch.zeros(self.n_cls, 1).long().to(self.text_feat.device)
            self.first_flag = False

        return self.text_feat, self.text_feat_full

        # text_features = []
        # prompts = self.prompt_learner(with_std=True)  ## torch.Size([1000, 77, 512])
        # tokenized_prompts = self.prompt_learner.tokenized_prompts
        # t_features = self.text_encoder(prompts, tokenized_prompts)  ## torch.Size([1000, 1024])
        # text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        # self.num_class = t_features.size(0)
        # text_features = torch.stack(text_features, dim=0)
        # # return text_features
        #
        # return torch.mean(text_features, dim=0)
    
    def DMN_encode_image(self, x):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  ## torch.Size([128, 197, 768])

        # ipdb.set_trace()
        # x = self.clip.visual.ln_post(x[:, 0, :]) ## 128*768
        x = self.clip.visual.ln_post(x) ## 128*197*768

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x
    def get_image_features(self, image):
        # image_features_vanilla = self.image_encoder(image.type(self.dtype))
        ## for Res50 128*1024 or 128*50*1024 [global feat; 7*7 local feature]
        ## for VIT,  128*512 or 128*197*512 [global feat; 14*14 local features]
        image_features = self.DMN_encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_local = image_features[:,1:,:]  ## B*L*C
        image_features_global = image_features[:, 0, :] ## B*C

        self.image_features_local = None #image_features_local
        self.image_features_global = image_features_global

        return self.image_features_global, self.image_features_local

        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()
        # return logits

    def forward(self, input):
        pass
        # if isinstance(input, Tuple):
        #     view_0, view_1, view_2 = input
        #     return self.contrast_prompt_tuning(view_0, view_1, view_2)
        # elif len(input.size()) == 2:
        #     return self.directional_prompt_tuning(input)
        # else:
        #     return self.inference(input)
