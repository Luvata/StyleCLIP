import torch
import clip

import torch.nn.functional as f
import torch.nn as nn
import numpy as np


class HookInputOutput:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.inputs.append(module_in)
        self.outputs.append(module_out)
        
    def clear(self):
        self.inputs = []
        self.outputs = []


def attention_pool(query, keys, temp=1):
    # print(query.shape, keys.shape)
    sim = query @ keys.T # (1, L)
    attn_w = f.softmax(sim / temp, dim=-1) # 1, L
    return attn_w @ keys # 1, L dot L, D = 1, D


class AttnPoolCLIP(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.model, self.preprocess = clip.load(
            opts.clip_model, 
            device="cpu", 
            download_root=opts.download_root
        )
        self.model.to(opts.device)
        self.is_vit = isinstance(self.model.visual, clip.model.VisionTransformer)
        self.hook = HookInputOutput()
        self.setup_visual_hook()

    def forward_visual_vit(self, image: torch.Tensor):
        assert self.hook_handler is not None
        assert self.is_vit

        img_embedding = self.model.visual(image) # (1, D)
        transformers_output = self.hook.outputs[0] # L, N, D
        x = self.model.visual.ln_post(
            transformers_output.permute(1, 0, 2)
        ) 
        x = x @ self.model.visual.proj # B, L, D

        assert np.allclose(torch.cosine_similarity(img_embedding, x[:, 0, :]).item(), 1)
        return img_embedding, x

    def forward_visual_resnet(self, image):
        assert self.hook_handler is not None
        assert not self.is_vit

        img_embedding = self.model.visual(image) # (1, D)
        inp = self.hook.inputs[0][0]
        x = inp.reshape(
            inp.shape[0], 
            inp.shape[1], 
            inp.shape[2] * inp.shape[3]
        ).permute(2, 0, 1) # N C H W => (HW) N C
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.model.visual.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = f.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.model.visual.attnpool.num_heads,
            q_proj_weight=self.model.visual.attnpool.q_proj.weight,
            k_proj_weight=self.model.visual.attnpool.k_proj.weight,
            v_proj_weight=self.model.visual.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([
                self.model.visual.attnpool.q_proj.bias,
                self.model.visual.attnpool.k_proj.bias, 
                self.model.visual.attnpool.v_proj.bias
            ]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.model.visual.attnpool.c_proj.weight,
            out_proj_bias=self.model.visual.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.model.visual.attnpool.training,
            need_weights=False
        )

        x = x.permute(1, 0, 2) # B, L, D
        assert np.allclose(torch.cosine_similarity(img_embedding, x[:, 0, :]).item(), 1)
        self.hook.clear()

        return img_embedding, x

    def setup_visual_hook(self):

        if self.is_vit:
            self.hook_handler = self.model.visual.transformer.register_forward_hook(self.hook)
            self.forward_visual = self.forward_visual_vit
        else:
            self.hook_handler = self.model.visual.attnpool.register_forward_hook(self.hook)
            self.forward_visual = self.forward_visual_resnet

    def forward(self, image, query):
        """Return query-pooled embedding
        image: tensor output of preprocess, shape [1, ...]
        query: tensor (1, D)
        """
        img_embedding, token_embeddings = self.forward_visual(image)
        token_embeddings = token_embeddings.squeeze(0) # remove batch dim 1
        return attention_pool(query, token_embeddings, temp=1)

