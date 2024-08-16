from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

def setup_rag(model, documents):
    if documents is None:
        print("No documents provided. RAG system will not be used.")
        return None

    # Create embeddings
    embeddings = OllamaEmbeddings(model="Losspost/stella_en_1.5b_v5")

    # Create vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    # Create retrieval chain
    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )