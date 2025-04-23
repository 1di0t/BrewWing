import os
import sys
from huggingface_hub import snapshot_download, HfApi

# list of model IDs to download
MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "facebook/nllb-200-distilled-600M",
    "sentence-transformers/all-MiniLM-L6-v2",
]

def verify_token():
    # Get token from environment variables
    hf_key = (
        os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )

    if not hf_key:
        print("Error: HUGGINGFACE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Verify token
    try:
        api = HfApi()
        user_info = api.whoami(token=hf_key)
        print(f"Authenticated as: {user_info}")
        return hf_key
    except Exception as e:
        print(f"Error verifying token: {str(e)}", file=sys.stderr)
        sys.exit(1)

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

if __name__ == "__main__":
    print("Starting model download process...")
    token = verify_token()
    
    for model_id in MODEL_IDS:
        download_model(model_id, token)
    
    print("All models downloaded successfully")