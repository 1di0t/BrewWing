from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStore
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate

def create_coffee_retrieval_qa_chain(llm: BaseLLM, vectorstore: VectorStore):
    """
    chain generation for coffee recommendation
    based on the LLM and vectorstore
    """
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt_template = (
        "당신은 커피 전문가입니다. 아래의 문맥을 참고하여 질문에 답변하세요.\n"
        "문맥: {context}\n"
        "질문: {question}\n"
        "답변:"
    )
    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # RetrievalQA chain generation
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs={"prompt": qa_prompt}
    )
    
    return qa_chain

