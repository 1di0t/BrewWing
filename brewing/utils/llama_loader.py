import os


def load_llama_llm(model_name_or_path="meta-llama/Llama-2-7b-hf", token=None):
    """
    - Llama2 model load (GPU environment recommended)
    - token: Hugging Face access token 
    """
    try:
        import torch._dynamo
        torch._dynamo.disable()
    except Exception as e:
        print(f"torch._dynamo.disable() failed: {e}")
        pass

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline

    cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/huggingface_cache")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        token=token,
        cache_dir=cache_dir, 
        local_files_only=True
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        token=token,
        cache_dir=cache_dir, 
        local_files_only=True,
        torch_dtype="auto",       
        device_map="auto",        
    )
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        max_new_tokens=128,
        truncation=True,
        temperature=0.7,
        top_p = 0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=generation_pipeline)
    return llm
