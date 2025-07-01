# handler.py
import runpod
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load once globally
logger.info("Starting to load FLUX.1-schnell model...")
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
    cache_dir="/workspace/hf_cache"
).to("cuda")
logger.info("FLUX.1-schnell model loaded successfully!")

def handler(event):
    try:
        logger.info(f"Received event: {event}")
        prompt = event.get("input", {}).get("prompt", "a surreal dreamscape")
        logger.info(f"Generating image with prompt: {prompt}")
        
        image = pipe(prompt=prompt).images[0]
        logger.info("Image generated successfully")

        # Convert to base64 string
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        base64_img = base64.b64encode(buffer.getvalue()).decode()
        logger.info("Image converted to base64 successfully")

        return {"image_base64": base64_img}
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {"error": str(e)}

# Start the serverless function
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
