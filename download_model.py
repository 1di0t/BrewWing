import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# list of model IDs to download
MODEL_IDS = [
    "meta-llama/Llama-3.2-1B",
    "facebook/nllb-200-distilled-600M",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# directory to cache the models
cache_dir = os.path.join(os.getcwd(), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

def download_model(model_id):
    try:
        print(f"Downloading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"Tokenizer for {model_id} downloaded.")

        print(f"Downloading model for {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"Model for {model_id} downloaded.")

        print(f"Downloading configuration for {model_id}...")
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"Configuration for {model_id} downloaded.")

    except Exception as e:
        print(f"Error downloading {model_id}: {e}")

if __name__ == "__main__":
    for model_id in MODEL_IDS:
        download_model(model_id)