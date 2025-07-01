# handler.py

from diffusers import DiffusionPipeline
import torch
import os
import uuid

# Cache dir (RunPod serverless uses /tmp or mounted volume)
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
    cache_dir=os.environ["HF_HOME"]
).to("cuda")

def handler(event):
    prompt = event.get("input", {}).get("prompt", "a surreal dreamscape")
    
    image = pipe(prompt=prompt).images[0]
    output_path = f"/tmp/{uuid.uuid4().hex}.png"
    image.save(output_path)
    
    return {
        "image_path": output_path
    }
