import re
import os

cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
model_path = os.path.join(cache_dir, "nllb-200-distilled-600M")

def extract_origin_text(data: str) -> str:
    """
    Extract the recommendation text from the result and clean any repetitive patterns
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
    
    extracted_text = ""
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, data, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            logger.info(f"Pattern {i+1} matched. Extracted {len(extracted)} characters")
            extracted_text = extracted
            break
    
    if not extracted_text:
        logger.warning("All extraction patterns failed. Returning original text.")
        extracted_text = data
        
    # 마이너스 기호(-) 반복 제거
    clean_text = re.sub(r'-{3,}', '---', extracted_text)
    clean_text = re.sub(r'(coffee bean\s*-\s*)+coffee bean', 'coffee bean', clean_text, flags=re.IGNORECASE)
    
    # 기타 반복 패턴 정리
    repeated_phrases = [
        r'(coffee\s+bean\s*)+',
        r'(\s*-\s*)+',
        r'(\*\s*)+',
        r'(#\s*)+',
    ]
    
    for pattern in repeated_phrases:
        before_length = len(clean_text)
        clean_text = re.sub(pattern, '\1', clean_text, flags=re.IGNORECASE)
        after_length = len(clean_text)
        if before_length != after_length:
            logger.info(f"Cleaned repeated pattern: {pattern}, removed {before_length - after_length} characters")
    
    # 정리된 내용이 너무 짧으면 원본 사용
    if len(clean_text) < len(extracted_text) * 0.5:
        logger.warning(f"Cleaned text too short ({len(clean_text)} vs {len(extracted_text)}). Using original.")
        return extracted_text
    
    logger.info(f"Text cleaning completed. Final length: {len(clean_text)}")
    return clean_text

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
