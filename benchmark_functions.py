'''
acc. to the paper following metrics were used:

1. LPIPS
2. CLIP-T
3. CLIP-I
4. DINO
5. DreamSim

'''

import lpips
import open_clip
import timm
import torch.nn.functional as F
import torchvision.transforms as T
from dreamsim import dreamsim
import torch

device = "cuda"

#LPIPS
lpips_model = lpips.LPIPS(net='vgg').eval().to(device)

def lpips_metric(src, edit):
    to_lpips_format = T.Compose([
        T.ToTensor(),               
        T.Normalize(
            mean=[0.5, 0.5, 0.5],    
            std=[0.5, 0.5, 0.5],
        ),
    ])

    t_src = to_lpips_format(src).unsqueeze(0).to(device)
    t_edit = to_lpips_format(edit).unsqueeze(0).to(device)

    with torch.no_grad():
        dist = lpips_model(t_src, t_edit)
    
    return dist.item()

#CLIP - helpers and main function
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model = clip_model.eval().to(device)

clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

def clip_encode_img(img):
    img = clip_preprocess(img).unsqueeze(0).to(device)  
    with torch.no_grad():
        feat = clip_model.encode_image(img)  
    features = F.normalize(feat, dim=-1)     
    return features

def clip_encode_text(prompt):
    tokens = clip_tokenizer([prompt]).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens) 
    features = F.normalize(feat, dim=-1)
    return features

def CLIP_T_metric(edit_img, edit_prompt): #call ts
    #so we're finding the cosine similarity between the target prompt and tgt img
    img_ft = clip_encode_img(edit_img)
    prompt_ft = clip_encode_text(edit_prompt)

    sim = (img_ft @ prompt_ft.T).item()
    return sim

def CLIP_I_metric(src_img, tgt_img): #call ts
    #sim b/w src and edit
    src_ft = clip_encode_img(src_img)
    tgt_ft = clip_encode_img(tgt_img)
    sim = (src_ft @ tgt_ft.T).item()
    return sim

#DINO
dino_model = timm.create_model(
    "vit_small_patch16_224_dino",
    pretrained=True
).eval().to(device)

dino_preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

def dino_metric(src_img, edit_img):
    src = dino_preprocess(src_img).unsqueeze(0).to(device)
    edit = dino_preprocess(edit_img).unsqueeze(0).to(device)

    with torch.no_grad():
        ft_src = dino_model(src)
        ft_edit = dino_model(edit)
    
    ft_src = F.normalize(ft_src, dim=-1)
    ft_edit = F.normalize(ft_edit, dim=-1)

    sim = (ft_src @ ft_edit.T).item()
    return sim

#dreamsim
dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True)
dreamsim_model = dreamsim_model.eval().to(device)

def dreamsim_metric(src_img, edit_img):
    src = dreamsim_preprocess(src_img).unsqueeze(0).to(device)
    edit = dreamsim_preprocess(edit_img).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = dreamsim_model(src, edit)

    return dist.item()