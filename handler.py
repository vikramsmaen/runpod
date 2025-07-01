# handler.py
import runpod
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the pipeline
pipe = None

def load_model():
    """Load the model lazily on first request"""
    global pipe
    if pipe is None:
        logger.info("Loading FLUX.1-schnell model for the first time...")
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.error("CUDA is not available!")
                raise RuntimeError("CUDA not available")
            
            logger.info(f"CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            
            # Load with optimizations
            pipe = DiffusionPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.float16,
                cache_dir="/workspace/hf_cache",
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # Move to CUDA with memory optimization
            pipe = pipe.to("cuda")
            
            # Enable memory efficient attention if available
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers memory efficient attention enabled")
                except:
                    logger.info("xFormers not available, using default attention")
            
            logger.info("FLUX.1-schnell model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise e
    
    return pipe

def handler(event):
    try:
        logger.info(f"Received event: {event}")
        
        # Load model lazily on first request
        model_pipe = load_model()
        
        prompt = event.get("input", {}).get("prompt", "a surreal dreamscape")
        logger.info(f"Generating image with prompt: {prompt}")
        
        image = model_pipe(prompt=prompt).images[0]
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
