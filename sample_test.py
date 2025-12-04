import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from custom_sampler import FlowEditSampler 

print("Loading SD3 Pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    variant="fp16"  
)
pipe.to("cuda")

#DISABLE WHEN RUNNING ON RUNPOD/VAST.AI!
# pipe.enable_model_cpu_offload() #text encoder can stay on cpu

sampler = FlowEditSampler(pipe) #custom flow-edit sampler

# testing
image_path = "/scratch/pbk5339/flow-edit_implementation/cat.png" 
source_image = Image.open(image_path).convert("RGB").resize((1024, 1024))

# run
print("Running FlowEdit...")
result_image = sampler(
    source_img=source_image,
    source_prompt="A photo of a cat",  #  source
    target_prompt="A photo of a dog",  # target
    num_steps=50,
    n_max=25,   # start halfway (step 25)
    n_min=0,    # do it all the way
    cfg_src=1.5,
    cfg_target=5.5
)

# save
result_image.save("/scratch/pbk5339/flow-edit_implementation/result_dog.png")
print("Done! Saved to result_dog.png")