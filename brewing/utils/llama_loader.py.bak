import os
import logging
import traceback

# 로깅 설정
logger = logging.getLogger(__name__)

def load_llama_llm(model_name_or_path="meta-llama/Llama-3.2-1B", token=None):
    """
    - Llama2 model load (GPU environment recommended)
    - token: Hugging Face access token 
    """
    try:
        logger.info(f"Loading LLM model from: {model_name_or_path}")
        
        # 모델 경로 확인
        if os.path.exists(model_name_or_path):
            logger.info(f"Model directory exists at {model_name_or_path}")
            try:
                logger.info(f"Model directory contents: {os.listdir(model_name_or_path)}")
            except Exception as e:
                logger.error(f"Failed to list directory contents: {str(e)}")
        else:
            logger.warning(f"Model directory does not exist at {model_name_or_path}")
            # 대체 경로 시도
            alternative_paths = [
                "/app/huggingface_cache/Llama-3.2-1B",
                "./huggingface_cache/Llama-3.2-1B",
                "Llama-3.2-1B"
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found model at alternative path: {alt_path}")
                    model_name_or_path = alt_path
                    break
            else:
                logger.error("Model not found in any location")
        
        try:
            import torch
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA current device: {torch.cuda.current_device()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            import torch._dynamo
            torch._dynamo.disable()
            logger.info("torch._dynamo.disable() successful")
        except Exception as e:
            logger.warning(f"torch._dynamo.disable() failed: {e}")
            pass

        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import HuggingFacePipeline

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            token=token, 
            local_files_only=True
        )
        logger.info("Tokenizer loaded successfully")
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            token=token,
            local_files_only=True,
            torch_dtype="auto",       
            device_map="auto",        
        )
        logger.info("Model loaded successfully")
        
        logger.info("Creating generation pipeline...")
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            truncation=True,
            temperature=0.7,
            top_p = 0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info("Generation pipeline created successfully")
        
        llm = HuggingFacePipeline(pipeline=generation_pipeline)
        logger.info("LLM initialized successfully")
        
        return llm
    except Exception as e:
        logger.error(f"Error in load_llama_llm: {str(e)}")
        logger.error(traceback.format_exc())
        raise
