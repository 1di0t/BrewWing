import os
import logging
import requests
import time
import traceback
from typing import Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)

# Hugging Face API 설정
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "YOUR_HF_API_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3-8B-Instruct"

class HuggingFaceAPI:
    """
    Hugging Face Inference API를 사용하여 텍스트 생성을 처리하는 클래스
    """
    
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        if HF_API_TOKEN == "YOUR_HF_API_TOKEN" or not HF_API_TOKEN:
            logger.warning("Hugging Face API token is not properly set. API calls will not work correctly.")
        else:
            logger.info("HuggingFaceAPI initialized with API token")
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """
        주어진 프롬프트에 대해 Hugging Face API를 통해 텍스트를 생성합니다.
        
        Args:
            prompt: 입력 프롬프트
            max_tokens: 생성할 최대 토큰 수
            temperature: 생성 다양성 조절 파라미터
            
        Returns:
            생성된 텍스트
        """
        try:
            logger.info(f"Sending request to Hugging Face API, prompt length: {len(prompt)}")
            start_time = time.time()
            
            # API 요청 설정
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.85,
                    "repetition_penalty": 1.2,
                    "return_full_text": False
                }
            }
            
            # API 호출
            response = requests.post(HF_API_URL, headers=self.headers, json=payload)
            
            # 응답 시간 로깅
            generation_time = time.time() - start_time
            logger.info(f"API response received in {generation_time:.2f} seconds, status: {response.status_code}")
            
            # 응답 처리
            if response.status_code == 200:
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    raw_text = result[0].get("generated_text", str(result[0]))
                else:
                    raw_text = str(result)
                
                logger.info(f"API response length: {len(raw_text)}")
                return raw_text
                
            elif response.status_code == 503 and "loading" in response.text.lower():
                # 모델 로딩 중일 경우 재시도
                logger.warning("Model is still loading, waiting and retrying...")
                time.sleep(10)
                
                # 재시도
                retry_response = requests.post(HF_API_URL, headers=self.headers, json=payload)
                if retry_response.status_code == 200:
                    result = retry_response.json()
                    if isinstance(result, list) and len(result) > 0:
                        raw_text = result[0].get("generated_text", str(result[0]))
                        return raw_text
                    
                logger.error(f"Retry failed: {retry_response.status_code} - {retry_response.text}")
                return ""
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            logger.error(traceback.format_exc())
            return ""