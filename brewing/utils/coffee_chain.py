from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStore
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate

def create_coffee_retrieval_qa_chain(llm: BaseLLM, vectorstore: VectorStore):
    """
    chain generation for coffee recommendation
    based on the LLM and vectorstore
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # 벡터 스토어 확인
    logger.info(f"Vector store type: {type(vectorstore)}")
    
    # 검색 패러미터 - 다양한 원두 추천을 위해 검색 결과 수 증가
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 9})
    logger.info("Retriever created successfully")

    # 다양한 원두 추천을 위한 프롬프트 수정
    template = """
    당신은 전문 커피 추천 시스템입니다. 다양한 원두의 커피를 추천해주는 역할을 합니다.
    주어진 정보를 사용하여 사용자 질문에 맞는 커피를 추천해주세요.
    
    사용자가 선호하는 맛이나 특징이 무엇인지 분석하고, 가능한 다양한 원두와 지역의 커피를 추천해주세요.
    
    중요: 절대로 다시(-) 기호를 반복해서 사용하지 마세요. 다시 기호는 불필요하며 중복해서 사용하면 안됩니다.
    
    이해하기 쪽한 형식으로, 다음 예시처럼 각 원두도 번호로 구분하여 추천해주세요:
    
    1. [원산지] [커피이름]: 맛 프로필, 로스팅 레벨, 특징
    2. [원산지] [커피이름]: 맛 프로필, 로스팅 레벨, 특징
    3. [원산지] [커피이름]: 맛 프로필, 로스팅 레벨, 특징
    
    ##참고 정보
    {context}
    
    ##질문
    {question}
    
    ##추천
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    logger.info("Prompt created successfully")

    # stuff 체인 타입 생성
    logger.info("Creating RetrievalQA chain with stuff type...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 변경: map_reduce → stuff
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
        },
        verbose=False
    )
    
    logger.info("QA chain created successfully")
    return qa_chain