import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def load_documents():
    documents = []
    context_dir = os.path.join(os.path.dirname(__file__), '..', 'context')
    
    if not os.path.exists(context_dir):
        print("Context directory not found. Creating it...")
        os.makedirs(context_dir)
    
    file_count = 0
    for filename in os.listdir(context_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(context_dir, filename)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            file_count += 1

    if file_count == 0:
        print("No text files found in the context directory. RAG will not be used.")
        return None

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)