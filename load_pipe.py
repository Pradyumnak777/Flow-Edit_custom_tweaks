import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from custom_sampler import FlowEditSampler 

def load_sd3_pipe():
    print("Loading SD3 Pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16,
        variant="fp16"  
    )
    pipe.to("cuda")

    #DISABLE WHEN RUNNING ON RUNPOD/VAST.AI!
    # pipe.enable_model_cpu_offload() #text encoder can stay on cpu

    return FlowEditSampler(pipe) #custom flow-edit sampler