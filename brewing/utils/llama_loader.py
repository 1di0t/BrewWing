import os
import logging
import traceback
import torch

# 로깅 설정
logger = logging.getLogger(__name__)

def load_llama_llm(model_name_or_path="meta-llama/Llama-3.2-1B", token=None):
    """
    - Llama 모델 로드 (직접 호출 방식으로 최적화)
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
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA current device: {torch.cuda.current_device()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            # torch._dynamo 비활성화 (성능 최적화)
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
        # 작은 모델에 최적화된 설정
        model_kwargs = {
            "token": token,
            "local_files_only": True,
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        
        # GPU 메모리가 제한적이라면 8비트 양자화 시도
        try:
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                logger.info(f"Free GPU memory: {free_memory / 1024 / 1024:.2f} MB")
                
                # 사용 가능한 메모리가 4GB 미만이면 8비트 양자화 적용
                if free_memory < 4 * 1024 * 1024 * 1024:
                    logger.info("Limited GPU memory detected. Using 8-bit quantization.")
                    model_kwargs["load_in_8bit"] = True
        except Exception as e:
            logger.warning(f"Failed to check GPU memory: {e}")
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        logger.info("Model loaded successfully")
        
        logger.info("Creating generation pipeline...")
        # 최적화된 파이프라인 설정
        generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            load_in_4bit=True,
            max_new_tokens=100,  # 더 긴 응답 허용
            truncation=True,
            temperature=0.5,     # 낮은 온도로 더 집중된 응답
            top_p=0.9,
            repetition_penalty=1.2,  # 반복 방지
            pad_token_id=tokenizer.eos_token_id,
            # 작은 모델을 위한 최적화 설정
            do_sample=True,  # 다양성을 위한 샘플링
            no_repeat_ngram_size=2,  # 반복 방지
            return_full_text=False,  # 프롬프트 반복 방지
        )
        logger.info("Generation pipeline created successfully")
        
        llm = HuggingFacePipeline(pipeline=generation_pipeline)
        logger.info("LLM initialized successfully")
        
        return llm
    except Exception as e:
        logger.error(f"Error in load_llama_llm: {str(e)}")
        logger.error(traceback.format_exc())
        raise
