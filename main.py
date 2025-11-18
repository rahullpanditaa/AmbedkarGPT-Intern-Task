from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from .retrieval.search import Search


def main():
    repl_loop()



def repl_loop():
    while True:
        question = input("> ")
        if question.lower() == "exit":
            break
        response = rag_chain.invoke(question)
        print(response)


# invoke the llm
llm = OllamaLLM(model="mistral")

searcher = Search()
retriever = searcher.load_or_create_vector_db()

prompt = ChatPromptTemplate.from_template("""Answer the question based on the provided context,
    
    Question: {question}

Context: 
{context}

Provide an answer that addresses the question:""")

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

document_combiner = RunnableLambda(combine_docs)

rag_inputs = {
    "context": retriever | document_combiner,
    "question": RunnablePassthrough()
}

rag_chain = rag_inputs | prompt | llm
        
if __name__ == "__main__":
    main()
