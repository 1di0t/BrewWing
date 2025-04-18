import re
import os

cache_dir = os.getenv("HF_HOME", "/app/huggingface_cache")

def extract_origin_text(data: str) -> str:
    """
    Extract the origin text from the result text
    result_text: str
    """
    result_text = data

    # extract the origin text
    match = re.search(r"(\nOrigin:.*?\n\n)", result_text, re.DOTALL)

    if match:
        return match.group(2).strip()
    return ""

from transformers import pipeline

target_lang = 'kor_Hang'  # target language

translator = pipeline(
    'translation',
    model=cache_dir+"/nllb-200-distilled-600M",
    device=0,
    src_lang='eng_Latn',  # input language
    tgt_lang=target_lang,  # output language
    max_length=512,
    model_kwargs={"local_files_only": True}
)

def translate_with_linebreaks(text):
    """
    Translate text with line breaks
    text: str
    return: str
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    translated = translator(lines, batch_size=8)
    return '\n'.join([t['translation_text'] for t in translated])
