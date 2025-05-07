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
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    logger.info("Retriever created successfully")

    # 다양한 원두 추천을 위한 프롬프트 수정
    template = template = """
    커피 추천자로서 다음 정보를 바탕으로 사용자 질문에 맞는 커피를 추천해주세요:
    {context}

    질문: {question}

    3가지 이상의 커피를 다음 형식으로 추천해주세요:

    ## 커피 추천

    1. **[원산지] [커피이름]**
    - **맛**: 설명
    - **로스팅**: 레벨
    - **특징**: 설명

    2. **[원산지] [커피이름]**
    - **맛**: 설명
    - **로스팅**: 레벨
    - **특징**: 설명
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
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
        },
        verbose=False
    )
    
    logger.info("QA chain created successfully")
    return qa_chain