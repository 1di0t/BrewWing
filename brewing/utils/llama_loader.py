import os
os.environ["HF_HOME"] = "/app/huggingface_cache"


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

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token,)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",        
        torch_dtype="auto",       
        token=token
        
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
