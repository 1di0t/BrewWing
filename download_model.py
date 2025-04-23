import os
import sys
import re
import logging
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def clean_token(token):
    """Clean token from any potential Cloud Build formatting issues."""
    if not token:
        return None
    
    # If token starts with $ or $$, it's likely a variable name instead of the actual token
    if token.startswith('$'):
        logger.warning(f"Token appears to be a variable name: {token}")
        # Try to get the actual token from environment
        actual_token = os.getenv(token.lstrip('$'))
        if actual_token:
            logger.info(f"Successfully extracted token from environment variable")
            return actual_token
    
    # If token doesn't start with hf_, it's likely invalid
    if not token.startswith('hf_'):
        logger.warning(f"Token doesn't start with 'hf_', got: {token[:4]}")
    
    return token

def verify_token(token):
    """Verify the Hugging Face token."""
    if not token:
        logger.error("Token is None or empty")
        return False
        
    logger.debug(f"Verifying token...")
    logger.debug(f"Token length: {len(token)}")
    logger.debug(f"Token starts with: {token[:4]}")
    
    # Check if token starts with 'hf_'
    if not token.startswith('hf_'):
        logger.error(f"Token must start with 'hf_', got: {token[:4]}")
        return False
    
    # Check token length (should be at least 40 characters)
    if len(token) < 40:
        logger.error(f"Token length ({len(token)}) is too short")
        return False
    
    return True

def main():
    logger.info("Starting model download process...")
    
    # Get token from environment variables
    for env_var in ['HUGGINGFACE_API_KEY', 'HUGGINGFACE_HUB_TOKEN', 'HF_HUB_TOKEN', 'HF_API_KEY']:
        token = os.getenv(env_var)
        if token:
            logger.info(f"Found token in {env_var}")
            break
    
    # Debug information
    logger.debug(f"Raw token length: {len(token) if token else 0}")
    logger.debug(f"Raw token starts with: {token[:4] if token else 'None'}")
    
    # Clean the token
    token = clean_token(token)
    logger.debug(f"Cleaned token length: {len(token) if token else 0}")
    logger.debug(f"Cleaned token starts with: {token[:4] if token else 'None'}")
    
    # Print all environment variables
    logger.debug("All environment variables:")
    for var in ['HUGGINGFACE_API_KEY', 'HUGGINGFACE_HUB_TOKEN', 'HF_HUB_TOKEN', 'HF_API_KEY']:
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
        try:
            whoami = api.whoami()
            logger.info(f"Token verified successfully. Logged in as: {whoami}")
        except Exception as e:
            logger.error(f"Error during API verification: {str(e)}")
            sys.exit(1)
        
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