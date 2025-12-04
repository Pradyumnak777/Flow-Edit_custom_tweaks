import yaml
from pathlib import Path
from PIL import Image
from load_pipe import load_sd3_pipe
import os

full_path = Path("benchmarking/flowedit.yaml").resolve()
root_dir = full_path.parent
project_dir = root_dir.parent

with open("benchmarking/flowedit.yaml", "r") as f:
    data = yaml.safe_load(f)

for entry in data:
    img_path = (project_dir/ "benchmarking" / entry["init_img"]).resolve()
    source_img = Image.open(img_path).convert("RGB")

    source_prompt = entry["source_prompt"]

    for tgt_prompt, tgt_code in zip(entry["target_prompts"], entry["target_codes"]):
        print("running edit:", tgt_code)
        print("target prompt:", tgt_prompt)

        sampler = load_sd3_pipe()
        print("Running FlowEdit...")
        result_image = sampler( #ignoring negative prompts for now..
            source_img=source_img,
            source_prompt = source_prompt,  #  source
            target_prompt= tgt_prompt,  # target
            num_steps=50,
            n_max=33,   # start halfway (step 25)
            n_min=0,    # do it all the way
            cfg_src=3.5,
            cfg_target=13.5
        )

        # save using target code for benchmarking later
        # os.makedirs(f"/results/{tgt_code}", exist_ok=True)
        results_dir = Path.cwd() / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_image.save(results_dir / f"{tgt_code}.png")

                
