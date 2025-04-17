import os
from dotenv import load_dotenv
from django.conf import settings

from utils.data_processing import load_and_preprocess_coffee_data
from utils.vector_store import create_vector_store_from_coffee_df
from utils.llama_loader import load_llama_llm
from utils.coffee_chain import create_coffee_retrieval_qa_chain
from utils.text import extract_origin_text, translate_with_linebreaks

import numpy as np
np.zeros(1) 

# 환경 변수 로드 (Hugging Face API 키 등)
load_dotenv()

huggingface_token = os.getenv("HUGGINGFACE_API_KEY")


coffee_qa_chain = None

async def initialize_coffee_chain():
    """
    Initialize the Coffee QA Chain when the server starts.
    """
    global coffee_qa_chain

    DATA_FILE_PATH = os.path.join(settings.BASE_DIR, 'data', 'coffee_drop.csv')

    # 데이터 전처리
    coffee_df = load_and_preprocess_coffee_data(DATA_FILE_PATH)

    # 벡터 스토어 생성
    vectorstore = create_vector_store_from_coffee_df(coffee_df)

    # LLM 로드 (Hugging Face 모델)
    llm = load_llama_llm("meta-llama/Llama-3.2-1B", token=huggingface_token)

    # 체인 생성
    coffee_qa_chain = create_coffee_retrieval_qa_chain(llm, vectorstore)

async def recommend_coffee(query: str) -> dict:
    """
    Process user query and return the coffee recommendation.
    
    Args:
        query (str): User's input query.
    
    Returns:
        dict: Recommendation result.
    """
    global coffee_qa_chain

    if coffee_qa_chain is None:
        raise ValueError("Coffee QA Chain is not initialized. Call initialize_coffee_chain() first.")

    try:
        # 체인 실행 (질문 처리)
        answer = await coffee_qa_chain.invoke({"query": query})
    except Exception as e:
        print(f"Error during chain invocation: {e}")
        # 결과 텍스트 추출 및 번역 처리
        answer['result'] = await extract_origin_text(answer['result'])
        answer['result'] = await translate_with_linebreaks(answer['result'])

        return {"answer": answer}
