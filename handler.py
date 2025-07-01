# handler.py
import runpod
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Handler script starting...")
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

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
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Load with optimizations
            logger.info("Starting model loading from cache/download...")
            
            # Get HuggingFace token from environment (RunPod secret)
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                logger.error("HF_TOKEN not found in environment variables")
                logger.error("Available environment variables: " + str([k for k in os.environ.keys() if 'HF' in k.upper() or 'TOKEN' in k.upper()]))
                raise RuntimeError("HF_TOKEN is required for gated model access")
            else:
                logger.info("HF_TOKEN found from RunPod secret, using for authentication")
                logger.info(f"Token length: {len(hf_token)} characters")
                logger.info(f"Token starts with: {hf_token[:10]}...")  # Log first 10 chars for verification
            
            pipe = DiffusionPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.float16,
                cache_dir="/workspace/hf_cache",
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=hf_token
            )
            logger.info("Model loaded from disk, moving to CUDA...")
            
            # Move to CUDA with memory optimization
            pipe = pipe.to("cuda")
            logger.info("Model moved to CUDA successfully")
            
            # Enable memory efficient attention if available
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers memory efficient attention enabled")
                except Exception as xform_e:
                    logger.info(f"xFormers not available: {xform_e}")
            
            logger.info("FLUX.1-schnell model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
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
    logger.info("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
    logger.info("RunPod serverless handler started successfully")
