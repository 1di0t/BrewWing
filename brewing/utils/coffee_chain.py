from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStore
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate

def create_coffee_retrieval_qa_chain(llm: BaseLLM, vectorstore: VectorStore):
    """
    chain generation for coffee recommendation
    based on the LLM and vectorstore
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # create a question prompt to generate part answers
    question_template = """다음 정보를 참고하여, 이 부분 정보에만 기반해 질문에 답변하세요.
        [정보]
        {context}
        [질문]
        {question}
        [부분 답변]"""
    question_prompt = PromptTemplate(
        template=question_template,
        input_variables=["context", "question"]
    )

    # combine answer and generate final answer
    combine_template = """여러 부분 답변을 종합해 최종 답변을 만드세요.
        [부분 답변들]
        {summaries}
        [질문]
        {question}
        [최종 답변]"""
    combine_prompt = PromptTemplate(
        template=combine_template,
        input_variables=["summaries", "question"]
    )

    # RetrievalQA chain generation
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "combine_prompt": combine_prompt,
        },
        verbose=True 
    )
    
    return qa_chain

