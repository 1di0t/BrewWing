import os
import sys
import logging
from huggingface_hub import snapshot_download, HfApi
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of model IDs to download
MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "facebook/nllb-200-distilled-600M",
    "sentence-transformers/all-MiniLM-L6-v2",
]

def main():
    logger.info("Starting model download process...")
    
    # Get token from environment variables (try different variable names)
    token = None
    for env_var in ['HUGGINGFACE_API_KEY', 'HUGGINGFACE_HUB_TOKEN', 'HF_HUB_TOKEN']:
        token = os.environ.get(env_var)
        if token:
            logger.info(f"Found token in {env_var}")
            # Print minimal debug info without exposing the token
            logger.info(f"Token length: {len(token)}")
            logger.info(f"Token starts with: {token[:4] if len(token) >= 4 else 'too short'}")
            break
    
    # Skip dummy token if present
    if token and ("dummy_token" in token or "testing_purposes" in token):
        logger.warning("Detected dummy token, will try public downloads only")
        token = None
    
    if not token:
        logger.warning("No valid Hugging Face token found, will try to download public models only")
    else:
        # Basic token validation
        if not token.startswith('hf_'):
            logger.warning(f"Token should start with 'hf_', got: {token[:4] if len(token) >= 4 else 'too short'}")
            # Continue anyway
        
        # Verify token with API if available
        try:
            api = HfApi(token=token)
            whoami = api.whoami()
            logger.info(f"Token verified successfully. Logged in as: {whoami['name']}")
        except Exception as e:
            logger.warning(f"Token verification warning: {str(e)}")
            logger.warning("Will try to continue with downloads anyway")
    
    # Download models
    success_count = 0
    failed_models = []
    
    for model_id in MODEL_IDS:
        try:
            logger.info(f"Downloading model: {model_id}")
            
            # Set model download directory
            cache_dir = os.environ.get("HF_HOME", "/app/huggingface_cache")
            local_dir = os.path.join(cache_dir, model_id.split("/")[-1])
            
            # Try to download the model
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                token=token
            )
            
            logger.info(f"Successfully downloaded model: {model_id}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {str(e)}")
            logger.error(traceback.format_exc())
            failed_models.append(model_id)
            
            # Create placeholder for failed model
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, "DOWNLOAD_FAILED.txt"), "w") as f:
                f.write(f"Model download failed for {model_id}\n")
    
    # Summary
    logger.info(f"Download summary: {success_count} successful, {len(failed_models)} failed")
    if failed_models:
        logger.warning(f"Failed models: {', '.join(failed_models)}")
    else:
        logger.info("All models downloaded successfully")
    
    # Always exit with success to allow build to continue
    sys.exit(0)

if __name__ == "__main__":
    main()