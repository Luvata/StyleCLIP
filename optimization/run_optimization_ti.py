import argparse
import math
import os

import torch
import torchvision
from torch import optim
import torch.nn.functional as f
from tqdm import tqdm

from PIL import Image
from criteria.id_loss import IDLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator

import clip
from utils import ensure_checkpoint_exists
from torch.cuda.amp import autocast

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]

imagenet_templates = [
    'a bad photo of a {}.',
#    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


def avg_text_embedding(txt, templates, model, device):
    with torch.no_grad():
        texts = [template.format(txt) for template in templates] #format with class
        texts = clip.tokenize(texts).to(device) #tokenize
        class_embeddings = model.encode_text(texts) #embed with text encoder
        class_embedding = class_embeddings.mean(dim=0)
    return class_embedding


def vit_forward_keep_tokens_embedding(model, x):
    x = model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    xs = x.permute(1, 0, 2)  # LND -> NLD
    xs = model.ln_post(xs)  # LND
    if model.proj is not None:
        xs = xs @ model.proj
    return xs[:, 0, :], xs # first token and all tokens


def resnet_forward_keep_tokens_embedding(model, x):
    def stem(x):
        for conv, bn in [(model.conv1, model.bn1), (model.conv2, model.bn2), (model.conv3, model.bn3)]:
            x = model.relu(bn(conv(x)))
        x = model.avgpool(x)
        return x

    x = x.type(model.conv1.weight.dtype)
    x = stem(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    # We need to modify from here
    # x = model.attnpool(x)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
    x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
    x = x + model.attnpool.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
    x, _ = F.multi_head_attention_forward(
        query=x, key=x, value=x,
        embed_dim_to_check=x.shape[-1],
        num_heads=model.attnpool.num_heads,
        q_proj_weight=model.attnpool.q_proj.weight,
        k_proj_weight=model.attnpool.k_proj.weight,
        v_proj_weight=model.attnpool.v_proj.weight,
        in_proj_weight=None,
        in_proj_bias=torch.cat([model.attnpool.q_proj.bias, model.attnpool.k_proj.bias, model.attnpool.v_proj.bias]),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0,
        out_proj_weight=model.attnpool.c_proj.weight,
        out_proj_bias=model.attnpool.c_proj.bias,
        use_separate_proj_weight=True,
        training=model.attnpool.training,
        need_weights=False
    )

    return x[0], x # x shape HW+1 N D


def attention_pool(query, keys, temp=1):
    # print(query.shape, keys.shape)
    sim = query @ keys.T # (1, L)
    attn_w = f.softmax(sim / temp, dim=-1) # 1, L
    return attn_w @ keys # 1, L dot L, D = 1, D
        

def extract_attr_embed(txt, img_path, model, preprocess, device):
    # pass
    # 1. Calcuate txt embedding
    # txt_embedding = avg_text_embedding(txt, imagenet_templates, model, device).unsqueeze(0) # (D)

    with torch.no_grad():
        texts = clip.tokenize([txt]).to(device) #tokenize
        txt_embedding = model.encode_text(texts)[0] #embed with text encoder
    # 2. forward get tokens embedding
    # this code work for ViT for now, but ResNet is similar
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        if isinstance(model.visual, clip.model.VisionTransformer):
            _, tokens_embeddings = vit_forward_keep_tokens_embedding(model.visual, image)
        elif isinstance(mode.visual, clip.model.ModifiedResNet):
            _, tokens_embeddings = resnet_forward_keep_tokens_embedding(model.visual, image)

    return attention_pool(txt_embedding, tokens_embeddings[0])


def prep_img_for_clip(img_gen, stylegan_size):
    image = f.avg_pool2d(
        f.upsample(img_gen,scale_factor=7),
        kernel_size=stylegan_size // 32
    )
    return image


def clip_loss_with_attr(img_gen, stylegan_size, attr_emb, model, preprocess, device) :
    image = prep_img_for_clip(img_gen, stylegan_size)

    image_features = model.encode_image(image)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    normed_attr = attr_emb / attr_emb.norm()

    similarity = 1 - model.logit_scale.exp() * image_features @ normed_attr.T / 100
    return similarity.sum()


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def main(args):
    ensure_checkpoint_exists(args.ckpt)

    device = torch.device(args.device)
    clip_model = args.clip_model

    model, preprocess = clip.load(clip_model, device="cpu", download_root="/vinai/thanhlv19/workspace/clip")
    model = model.to(device)
    img_path = args.img_description
    attr = args.attr

    attr_embedding = extract_attr_embed(attr, img_path, model, preprocess, device).detach().clone()

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
        c_loss = clip_loss_with_attr(img_gen, args.stylegan_size, attr_embedding, model, preprocess, device)

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
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    parser.add_argument("--img_description", type=str, default="a person with purple hair", help="path to style image")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", choices=clip.available_models())
    parser.add_argument("--attr", type=str, default="hair", help="the attribute name")
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

