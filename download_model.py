import os
from huggingface_hub import snapshot_download

# list of model IDs to download
MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "facebook/nllb-200-distilled-600M",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# directory to cache the models
cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
hf_key = os.getenv("HUGGINGFACE_API_KEY")


# check token existence
if not hf_key:
    raise RuntimeError("HUGGINGFACE_API_KEY environment variable is not set.")

def download_model(model_id):
    local_dir = os.path.join(cache_dir, model_id.split("/")[-1])
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        local_files_only=False,
        token=hf_key,
    )

if __name__ == "__main__":
    for model_id in MODEL_IDS:
        download_model(model_id)