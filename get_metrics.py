from benchmark_functions import lpips_metric, CLIP_I_metric, CLIP_T_metric, dino_metric, dreamsim_metric
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
#run a loop over the results folder, compare every src and tgt images (also prompts for clip-t)

full_path = Path("benchmarking/flowedit.yaml").resolve()
root_dir = full_path.parent
project_dir = root_dir.parent

with open("benchmarking/flowedit.yaml", "r") as f:
    data = yaml.safe_load(f)


global_benchmarks = { #for averaging later
    "lpips": [],
    "clip_t": [],
    "clip_i": [],
    "dino": [],
    "dreamsim": []
}


for entry in data:
    #GATHER DATA FROM HERE!

    #get the source img path first and load it
    src_img_path = (project_dir/ "benchmarking" / entry["init_img"]).resolve()
    source_img = Image.open(src_img_path).convert("RGB")

    #get the paths of all the tgt images for this entry
    tgt_imgs_paths = []
    for target_code in entry["target_codes"]:
        tgt_imgs_path = (project_dir/ "results" / Path(entry["init_img"]).stem / f"{target_code}.png").resolve()
        tgt_imgs_paths.append(tgt_imgs_path)
    
    #now get the corresponding target prompts (for clip-t)
    tgt_prompts = entry["target_prompts"]


    #compute metrics for this entry!
    for target_path, target_prompt in zip(tgt_imgs_paths, tgt_prompts):
        target_img = Image.open(target_path).convert("RGB")

        #LPIPS metric-
        lpips_val = lpips_metric(source_img, target_img)
        global_benchmarks["lpips"].append(lpips_val)

        #CLIP-T
        clip_t_val = CLIP_T_metric(target_img, target_prompt)
        global_benchmarks["clip_t"].append(clip_t_val)

        #CLIP-I
        clip_i_val = CLIP_I_metric(source_img, target_img)
        global_benchmarks["clip_i"].append(clip_i_val)

        #DINO
        dino_val = dino_metric(source_img, target_img)
        global_benchmarks["dino"].append(dino_val)

        #dreamsim
        dreamsim_val = dreamsim_metric(source_img, target_img)
        global_benchmarks["dreamsim"].append(dreamsim_val)

        print(f"metrics done for {target_path.name}")


#find global metrics
print("\n These are the final averaged benchmarks")
print("-"*35)

for metric, val in global_benchmarks.items():
    avg = np.mean(val) #
    print(f"{metric}: {avg}\n")

     