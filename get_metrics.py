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

device = "cuda"

#LPIPS
lpips_model = lpips.LPIPS(net='vgg').eval().to(device)

#CLIP
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
clip_model = clip_model.eval().to(device)

clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')