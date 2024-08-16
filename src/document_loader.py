import os
import re
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class BillTextSplitter(TextSplitter):
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text):
        # Hierarchical Splitting: Titles, Sections, Subsections
        titles = re.split(r'\n(?=TITLE [IVXLCDM]+)', text)
        
        chunks = []
        for title in titles:
            sections = re.split(r'\n(?=SECTION \d+\.)', title)
            for section in sections:
                if len(section) <= self._chunk_size:
                    chunks.append(section.strip())
                else:
                    # Further split within sections using regex for subsections
                    subsections = re.split(r'\n(?=Subtitle [A-Z]|CHAPTER \d+|Sec\.\s+\d+)', section)
                    for subsection in subsections:
                        if len(subsection) <= self._chunk_size:
                            chunks.append(subsection.strip())
                        else:
                            # Split on newlines if still too large
                            sub_chunks = self.split_on_newlines(subsection)
                            chunks.extend(sub_chunks)
        
        return chunks

    def split_on_newlines(self, text):
        # Split on newlines, respecting chunk_size and chunk_overlap
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            if current_length + len(line) > self._chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk).strip())
                overlap = self._chunk_overlap
                while overlap > 0 and current_chunk:
                    overlap -= len(current_chunk[-1]) + 1  # +1 for newline
                    current_chunk.pop(0)
                current_length = sum(len(l) for l in current_chunk) + len(current_chunk) - 1
            
            current_chunk.append(line)
            current_length += len(line) + 1  # +1 for newline
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk).strip())
        
        return chunks

def load_documents():
    documents = []
    context_dir = os.path.join(os.path.dirname(__file__), '..', 'context')
    
    if not os.path.exists(context_dir):
        print("Context directory not found. Creating it...")
        os.makedirs(context_dir)
    
    file_count = 0
    for filename in os.listdir(context_dir):
        file_path = os.path.join(context_dir, filename)
        if filename.endswith('.txt'):
            with open(file_path, 'r') as file:
                content = file.read()
            documents.append(Document(page_content=content, metadata={"source": filename}))
            file_count += 1
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            file_count += 1

    if file_count == 0:
        print("No text or PDF files found in the context directory. RAG will not be used.")
        return None

    bill_splitter = BillTextSplitter(chunk_size=5000, chunk_overlap=700)
    split_docs = []
    for doc in documents:
        split_docs.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in bill_splitter.split_text(doc.page_content)])
    
    return split_docs
