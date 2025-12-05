import yaml
from pathlib import Path
from PIL import Image
from load_pipe import load_sd3_pipe
import json
from datetime import datetime

full_path = Path("benchmarking/flowedit.yaml").resolve()
root_dir = full_path.parent
project_dir = root_dir.parent

with open("benchmarking/flowedit.yaml", "r") as f:
    data = yaml.safe_load(f)

sampler = load_sd3_pipe() #load this only once!

for entry in data:
    img_path = (project_dir/ "benchmarking" / entry["init_img"]).resolve()
    source_img = Image.open(img_path).convert("RGB")

    source_prompt = entry["source_prompt"]

    for tgt_prompt, tgt_code in zip(entry["target_prompts"], entry["target_codes"]):
        
        #extract img_name
        img_name = Path(entry["init_img"]).stem
        # save using target code for benchmarking later
        # os.makedirs(f"/results/{tgt_code}", exist_ok=True)
        results_dir = Path.cwd() / "results" / img_name
        results_dir.mkdir(parents=True, exist_ok=True)

        #checking if it exists (for crashes/resuming)
        out_path = results_dir / f"{tgt_code}.png"
        if out_path.exists():
            print(f"skipping! {tgt_code}.png already exists")
            continue
        
        print("running edit:", tgt_code)
        print("target prompt:", tgt_prompt)
        print("Running FlowEdit...")
        start_time = datetime.now()
        result_image = sampler( #ignoring negative prompts for now..
            source_img=source_img,
            source_prompt = source_prompt,  #  source
            target_prompt= tgt_prompt,  # target
            num_steps=50,
            n_max=33,   # 
            n_min=0,    # do it all the way
            cfg_src=3.5,
            cfg_target=13.5
        )
        end_time = datetime.now()
        
        result_image.save(results_dir / f"{tgt_code}.png")

        #creating metadata for reference
        metadata = {
            "source_prompt": source_prompt,
            "target_prompt": tgt_prompt,
            "target_code": tgt_code,
            "params_used": {
                "num_steps": 50,
                "n_max": 33,
                "n_min": 0,
                "cfg_src": 3.5,  #NEEDSA TO BE CHANGED MANUALLY FOR NOW, IF CHANGING THESE VALS!!
                "cfg_target": 13.5
            },
            "time_taken_seconds": str(end_time - start_time)
        }
        with open(results_dir / f"{tgt_code}.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
                
