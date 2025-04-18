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


def download_model(model_id):
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=None,
            local_dir_use_symlinks=False,
            local_files_only=False,
        )


    except Exception as e:
        print(f"Error downloading {model_id}: {e}")

if __name__ == "__main__":
    for model_id in MODEL_IDS:
        download_model(model_id)