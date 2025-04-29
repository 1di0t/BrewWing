import re
import os

cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")
model_path = os.path.join(cache_dir, "nllb-200-distilled-600M")
def extract_origin_text(data: str) -> str:
    """
    Extract the origin text from the result text
    result_text: str
    """
    result_text = data

    # extract the origin text
    match = re.search(r"(\nOrigin:.*?\n\n)", result_text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

target_lang = 'kor_Hang'  # target language

# 명시적으로 토크나이저와 모델을 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    
    # 모델과 토크나이저를 직접 전달하는 파이프라인 생성
    translator = pipeline(
        'translation',
        model=model,
        tokenizer=tokenizer,
        device=0,
        src_lang='eng_Latn',  # input language
        tgt_lang=target_lang,  # output language
        max_length=512
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
