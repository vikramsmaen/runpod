# handler.py
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO

# Load once globally
pipe = DiffusionPipeline.from_pretrained(
    "fluxml/flux-fast",
    torch_dtype=torch.float16,
    cache_dir="/workspace/hf_cache"
).to("cuda")

def handler(event):
    try:
        prompt = event.get("input", {}).get("prompt", "a surreal dreamscape")
        image = pipe(prompt=prompt).images[0]

        # Convert to base64 string
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode()

        return {"image_base64": base64_img}
    except Exception as e:
        return {"error": str(e)}
