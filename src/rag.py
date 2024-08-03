from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain

def setup_rag(model, documents):
    if documents is None:
        print("No documents provided. RAG system will not be used.")
        return None

    # Create embeddings
    embeddings = OllamaEmbeddings()

    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    # Create retrieval chain
    return create_retrieval_chain(
        llm=model,
        retriever=vector_store.as_retriever()
    )