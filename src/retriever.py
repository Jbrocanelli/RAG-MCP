from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def create_retrieval_chain(vectorstore, model_name: str = "llama-3.3-70b-versatile"):
    llm = ChatGroq(model_name=model_name, temperature=0.1)

    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template="""Use the context below to answer the question. If you don't know, say so.

        Context: {context}

        Question: {question}

        Answer:"""
    )

    """MMR(Maximal Marginal Relevance) balances relevance(how similar the chunk is to the query)
    and diversity(how diferent the chunk is from other already selected chunks)

    fetch_k is how many candidates it considers before picking the most diverse k"""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

    """the | operator pipes the output of each step as input to the next.
    RunnablePassThrough passes the unchanged original query as "question"
    eg. "context": chunks, "question": "what is attention"
    prompt takes the dict and fills the prompt template placeholders
    llm sends the formatted prompt to grok and gets back a ChatMessage object(not a string)
    StrOutputParser extracts the text content from the ChatMessage object and returns a plain string"""
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
