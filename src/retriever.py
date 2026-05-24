from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def create_retrieval_chain(vectorstore, model_name: str = "llama-3.3-70b-versatile"):
    llm = ChatGroq(model_name=model_name, temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise document assistant. Answer questions using only the provided context.
- Do not use outside knowledge under any circumstances.
- If the context lacks sufficient information, say: "I could not find an answer in the provided documents."
- If the context partially answers the question, state what you found and what is missing."""),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}")
    ])

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

    chain = RunnableParallel(
        answer=(
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        ),
        source_documents=retriever
    )

    return chain
