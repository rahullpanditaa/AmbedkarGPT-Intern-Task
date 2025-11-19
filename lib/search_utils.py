from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR_PATH = Path(__file__).parent.parent.resolve() / "data"
# SPEECH_TXT_PATH = DATA_DIR_PATH / "speech.txt"




CHROMA_DIR_PATH = Path(__file__).parent.parent.resolve() / "vector_db"

PROMPT = ChatPromptTemplate.from_template("""Answer the question based on the provided context,
    
Question: {question}

Context: 
{context}

Provide an answer that addresses the question:""")

def combine_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)