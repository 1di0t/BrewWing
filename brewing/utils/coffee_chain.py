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
    당신은 전문 커피 추천 시스템입니다. 사용자 질문에 따라 적합한 커피 원두를 추천해주세요.
    
    다음 정보를 참고하여 사용자에게 가장 적합한 커피를 추천해주세요:
    {context}
    
    사용자 질문: {question}
    
    지침:
    1. 사용자의 질문을 정확히 이해하고 분석하세요.
    2. 커피 가이드에서 분석한 정보를 바탕으로 최소 3가지 이상의 커피를 추천해주세요.
    3. 각 추천마다 원산지, 커피 이름, 맛 프로필, 로스팅 레벨, 특징을 설명해주세요.
    
    출력 형식:
    
    ## 커피 추천
    
    1. **[원산지] [커피이름]**
       - **맛 프로필**: (맛 설명)
       - **로스팅**: (로스팅 레벨)
       - **특징**: (특징 설명)
    
    2. **[원산지] [커피이름]**
       - **맛 프로필**: (맛 설명)
       - **로스팅**: (로스팅 레벨)
       - **특징**: (특징 설명)
    
    3. **[원산지] [커피이름]**
       - **맛 프로필**: (맛 설명)
       - **로스팅**: (로스팅 레벨)
       - **특징**: (특징 설명)
    
    반드시 이 형식을 지켜서 작성해주세요. 예시에 있는 괄호 문구는 제거하고 실제 내용을 작성해주세요.
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