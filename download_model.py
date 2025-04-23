import os
import sys
import logging
import json
import requests
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoTokenizer, AutoModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# List of model IDs to download
MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "facebook/nllb-200-distilled-600M",
    "sentence-transformers/all-MiniLM-L6-v2",
]

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
    
    # Remove any whitespace or newlines
    token = token.strip()
    
    return token

def verify_token_direct(token):
    """Directly verify the token with Hugging Face API using requests."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
        
        logger.debug(f"Direct API response status code: {response.status_code}")
        logger.debug(f"Direct API response headers: {json.dumps(dict(response.headers))}")
        
        if response.status_code == 200:
            logger.info("Token verified directly via API request")
            return True
        else:
            logger.error(f"Token verification failed with status code {response.status_code}")
            if response.text:
                logger.error(f"Response text: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error during direct token verification: {str(e)}")
        return False

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
    
    # Try direct verification as a more reliable method
    logger.debug("Attempting direct token verification...")
    direct_verification = verify_token_direct(token)
    
    if direct_verification:
        logger.info("Token verified successfully via direct API call")
    else:
        logger.warning("Token failed direct verification, trying with HfApi...")
    
    return True

def download_model(model_id, token):
    """Download a model from Hugging Face."""
    logger.info(f"Downloading model: {model_id}")
    try:
        # Set model download directory
        cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
        local_dir = os.path.join(cache_dir, model_id.split("/")[-1])
        
        # Try direct HTTP request first for public models
        try:
            logger.info(f"Trying HTTP download without token for {model_id}")
            model_url = f"https://huggingface.co/api/models/{model_id}"
            response = requests.get(model_url)
            logger.debug(f"Model info status code: {response.status_code}")
            
            # If model is publicly accessible, try downloading without token
            if response.status_code == 200:
                logger.info(f"Model {model_id} appears to be publicly accessible")
                
                try:
                    logger.info(f"Downloading {model_id} without token...")
                    snapshot_download(
                        repo_id=model_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        local_files_only=False
                    )
                    logger.info(f"Successfully downloaded model without token: {model_id}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to download without token: {str(e)}")
                    # Continue to token-based download
        except Exception as e:
            logger.warning(f"Error checking model public status: {str(e)}")
            # Continue to token-based download
        
        # Try download with token
        try:
            logger.info(f"Downloading {model_id} with token...")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                local_files_only=False,
                token=token
            )
            logger.info(f"Successfully downloaded model with token: {model_id}")
        except HfHubHTTPError as e:
            logger.error(f"HTTP error during download: {str(e)}")
            if "401 Client Error" in str(e):
                logger.error("Authentication error - token may be invalid or expired")
                # Try one more time with a direct request to see what's happening
                headers = {"Authorization": f"Bearer {token}"}
                model_files_url = f"https://huggingface.co/api/models/{model_id}/refs/main/tree"
                response = requests.get(model_files_url, headers=headers)
                logger.error(f"Direct model files request status: {response.status_code}")
                logger.error(f"Direct model files response: {response.text[:200]}")
            raise e
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        sys.exit(1)

def main():
    logger.info("Starting model download process...")
    
    # Print all environment variables for debugging
    logger.debug("All environment variables:")
    env_vars = os.environ.copy()
    for key, value in env_vars.items():
        if "TOKEN" in key.upper() or "API_KEY" in key.upper() or "HF_" in key.upper():
            value_display = value[:4] + "..." if value else "Not set"
            logger.debug(f"{key}: {value_display}")
    
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
            logger.error("Trying direct API call as alternative verification")
            
            # Try direct verification again as a fallback
            if verify_token_direct(token):
                logger.info("Token verified through direct API call")
            else:
                logger.error("Both verification methods failed")
                sys.exit(1)
        
        # Download models
        for model_id in MODEL_IDS:
            download_model(model_id, token)
        
        logger.info("All models downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error during model download: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()