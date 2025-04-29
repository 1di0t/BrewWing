import re
import os

cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
model_path = os.path.join(cache_dir, "nllb-200-distilled-600M")

def extract_origin_text(data: str) -> str:
    """
    Extract the recommendation text from the result
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 입력 데이터 로깅
    logger.info(f"Extracting text from data of length: {len(data)}")
    
    # 추천 섹션 추출 (다양한 패턴 시도)
    patterns = [
        r"##\s*\ucd94\ucc9c\s*\n(.+)",  # ##추천 뒤의 모든 텍스트
        r"\n\s*\ucd94\ucc9c[:]*\s*\n(.+)",  # "추천:" 뒤의 모든 텍스트
        r"\nOrigin:(.+?)\n\n",  # 기존 패턴
        r"\n\s*\ucd94\ucc9c[:]*\s*(.+)",  # "추천:" 뒤의 텍스트 (개행 없음)
        r"([\s\S]+)"  # 모든 텍스트 반환 (추출 실패 시 폴백)
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, data, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.info(f"Pattern {i+1} matched. Extracted {len(extracted)} characters")
            return extracted
    
    # 패턴 매칭 실패 시 원본 반환
    logger.warning("All extraction patterns failed. Returning original text.")
    return data

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

target_lang = 'kor_Hang'  # target language

# 명시적으로 토크나이저와 모델을 로드
try:
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Loading translation model from {model_path}")
    
    # CPU 로드 대신 GPU 사용 방지
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, 
        local_files_only=True,
        device_map="cpu",  # CPU로 지정
        low_cpu_mem_usage=True
    )
    
    # 모델과 토크나이저를 직접 전달하는 파이프라인 생성
    translator = pipeline(
        'translation',
        model=model,
        tokenizer=tokenizer,
        device="cpu",  # CPU로 지정
        src_lang='eng_Latn',  # input language
        tgt_lang=target_lang,  # output language
        max_length=512,
        batch_size=1  # 배치 사이즈 축소
    )
except Exception as e:
    import logging
    logging.error(f"Error loading translation model: {str(e)}")
    # Fallback to a dummy translator
    def translator(texts, batch_size=1):
        return [{"translation_text": text} for text in texts]


def translate_with_linebreaks(text):
    """
    Translate text with line breaks
    text: str
    return: str
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    translated = translator(lines, batch_size=8)
    return '\n'.join([t['translation_text'] for t in translated])
