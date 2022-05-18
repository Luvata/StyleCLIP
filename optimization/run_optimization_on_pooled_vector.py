import argparse
import math
import os

import torch
import torchvision

from torch import optim
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image

from tqdm import tqdm

from criteria.clip_loss import CLIPLoss
from criteria.id_loss import IDLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
import clip
from utils import ensure_checkpoint_exists
import numpy as np

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]


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
            device=opts.device, 
            download_root=opts.download_root
        )
        self.is_vit = isinstance(self.model.visual, clip.model.VisionTransformer)
        self.hook = HookInputOutput()
        self.setup_visual_hook()

    def setup_visual_hook(self):

        if self.is_vit:
            self.hook_handler = self.model.visual.transformer.register_forward_hook(self.hook)
            self.forward_visual = self.vit_forward_visual
        else:
            self.hook_handler = self.model.visual.attnpool.register_forward_hook(self.hook)
            self.forward_visual = self.resnet_forward_visual


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
        x, _ = F.multi_head_attention_forward(
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

        return img_embedding, x
        

    def forward(self, image, query):
        """Return query-pooled embedding
        image: tensor output of preprocess, shape [1, ...]
        query: tensor (1, D)
        """
        img_embedding, token_embeddings = self.forward_visual(image)
        token_embeddings = token_embeddings.squeeze(0) # remove batch dim 1
        self.hook.clear()
        return attention_pool(query, token_embeddings, temp=1)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    ensure_checkpoint_exists(args.ckpt)

    device = torch.device(args.device)
    text_inputs = torch.cat([clip.tokenize(args.attr)]).to(device)

    ap_clip = AttnPoolCLIP(args)

    # 1. Get attribute vector
    style_image = ap_clip.preprocess(Image.open(args.img_description)).unsqueeze(0).to(device)

    texts = clip.tokenize([args.attr]).to(device)
    attr_embedding = ap_clip.model.encode_text(texts) # (1, D)

    with torch.no_grad():
        attr_img_embedding = ap_clip.forward(style_image, attr_embedding) # 1, D
        attr_img_embedding = attr_img_embedding / attr_img_embedding.norm(dim=1, keepdim=True)

    ##################################

    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    mean_latent = g_ema.mean_latent(4096)

    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).to(device)
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).to(device)
        with torch.no_grad():
            _, latent_code_init, _ = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    if args.work_in_stylespace:
        with torch.no_grad():
            _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
        latent = [s.detach().clone() for s in latent_code_init]
        for c, s in enumerate(latent):
            if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                s.requires_grad = True
    else:
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

    id_loss = IDLoss(args)

    if args.work_in_stylespace:
        optimizer = optim.Adam(latent, lr=args.lr)
    else:
        optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

        # c_loss = clip_loss(img_gen, text_inputs)
        ## Step 2, pass img_gen through preprocess, a.k.a avgpool2d and upsample
        img_gen = f.avg_pool2d(
            f.upsample(img_gen,scale_factor=7),
            kernel_size=args.stylegan_size // 32
        )
        ## Step 3, calculate pooled embedding between img_gen and attr_embedding
        gen_attr_img_embedding = ap_clip.forward(img_gen, attr_embedding)
        ## Step 4, get cosine loss
        gen_attr_img_embedding = gen_attr_img_embedding / gen_attr_img_embedding.norm(dim=1, keepdim=True)
        c_loss = 1 - ap_clip.model.logit_scale.exp() * gen_attr_img_embedding @ attr_img_embedding.T / 100
        c_loss = c_loss.sum()

        if args.id_lambda > 0:
            i_loss = id_loss(img_gen, img_orig)[0]
        else:
            i_loss = 0

        if args.mode == "edit":
            if args.work_in_stylespace:
                l2_loss = sum([((latent_code_init[c] - latent[c]) ** 2).sum() for c in range(len(latent_code_init))])
            else:
                l2_loss = ((latent_code_init - latent) ** 2).sum()
            loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss
        else:
            loss = c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

            torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.jpg", normalize=True, range=(-1, 1))

    if args.mode == "edit":
        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen

    return final_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Cuda device")
    parser.add_argument("--img_description", type=str, default="./jk.jpg", help="path to style image")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", choices=clip.available_models())
    parser.add_argument("--attr", type=str, default="hair", help="the attribute name")
    parser.add_argument("--download_root", type=str, default=None, help="path to CLIP models")

    # original params #
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"], help="choose between edit an image an generate a free one")
    parser.add_argument("--l2_lambda", type=float, default=0.008, help="weight of the latent distance (used for editing only)")
    parser.add_argument("--id_lambda", type=float, default=0.000, help="weight of id loss (used for editing only)")
    parser.add_argument("--latent_path", type=str, default=None, help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                                                                      "the mean latent in a free generation, and from a random one in editing. "
                                                                      "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7, help="used only for the initial latent vector, and only when a latent code path is"
                                                                      "not provided")
    parser.add_argument('--work_in_stylespace', default=False, action='store_true')
    parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--ir_se50_weights', default='../pretrained_models/model_ir_se50.pth', type=str,
                             help="Path to facial recognition network used in ID loss")

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"), normalize=True, scale_each=True, range=(-1, 1))


