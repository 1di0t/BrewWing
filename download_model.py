import os
import sys
from huggingface_hub import snapshot_download, HfApi
from transformers import AutoTokenizer, AutoModel
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# list of model IDs to download
MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "facebook/nllb-200-distilled-600M",
    "sentence-transformers/all-MiniLM-L6-v2",
]

def verify_token(token):
    """Verify the Hugging Face token."""
    logger.debug(f"Verifying token...")
    logger.debug(f"Token length: {len(token)}")
    logger.debug(f"Token starts with: {token[:4]}")
    
    # Check if token starts with 'hf_'
    if not token.startswith('hf_'):
        logger.error("Token must start with 'hf_'")
        return False
    
    # Check token length (should be at least 40 characters)
    if len(token) < 40:
        logger.error(f"Token length ({len(token)}) is too short")
        return False
    
    return True

def download_model(model_id, token):
    print(f"Downloading model: {model_id}")
    try:
        cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
        local_dir = os.path.join(cache_dir, model_id.split("/")[-1])
        
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            local_files_only=False,
            use_auth_token=token,
        )
        print(f"Successfully downloaded model: {model_id}")
    except Exception as e:
        print(f"Error downloading model {model_id}: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    logger.info("Starting model download process...")
    
    # Get token from environment variables
    token = os.getenv('HUGGINGFACE_API_KEY') or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_HUB_TOKEN')
    
    # Debug information
    logger.debug(f"Token length: {len(token) if token else 0}")
    logger.debug(f"Token starts with: {token[:4] if token else 'None'}")
    
    # Print all environment variables
    logger.debug("All environment variables:")
    for var in ['HUGGINGFACE_API_KEY', 'HUGGINGFACE_HUB_TOKEN', 'HF_HUB_TOKEN']:
        logger.debug(f"{var}: {'Set' if os.getenv(var) else 'Not set'}")
    
    if not token:
        logger.error("No Hugging Face token found in environment variables")
        sys.exit(1)
    
    # Verify token
    if not verify_token(token):
        logger.error("Token validation failed")
        sys.exit(1)
    
    try:
        # Create HfApi instance
        logger.debug("Creating HfApi instance with token")
        api = HfApi(token=token)
        
        # Verify token with API call
        logger.debug("Verifying token with API call")
        api.whoami()
        logger.info("Token verified successfully")
        
        # Download model
        logger.info("Downloading model...")
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="/app/huggingface_cache",
            token=token
        )
        logger.info("Model downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error during model download: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()