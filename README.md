# llm_bill_analyzer

POC to have LLM summarize congressional bills.

## LangChain RAG Project with Ollama

This project demonstrates the use of LangChain with Ollama for Retrieval-Augmented Generation (RAG) in a chat-based interface. It allows users to interact with an AI assistant that leverages local context files to provide informed responses.

### Features

- Integration with Ollama for local language model execution
- Retrieval-Augmented Generation (RAG) using local context files
- Chat-based interface for continuous interaction with the AI
- Customizable prompt engineering
- Efficient document loading and processing

### Project Structure

```markdown
llm_bill_analyzer/
│
├── context/
│   └── (relevant english text files)
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── document_loader.py
│   ├── rag.py
│   └── prompts.py
│
├── requirements.txt
└── README.md
```

- `context/`: Directory for storing text files used as context for RAG
- `src/`: Source code directory
  - `main.py`: Entry point of the application, handles the chat loop
  - `model.py`: Ollama model setup
  - `document_loader.py`: Loads and processes documents
  - `rag.py`: Sets up the RAG pipeline
  - `prompts.py`: Defines the chat prompt template
- `requirements.txt`: Lists project dependencies

### Prerequisites

- Python 3
- Ollama installed and running locally.

### Installation

1. Pull llama 3.1:

   ```bash
    ollama run llama3.1
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Place your context text files in the `context/` folder. These files will be used by the RAG system to provide relevant information to the AI.

2. Run the main script:

   ```bash
   python3 src/main.py
   ```

3. Start chatting with the AI assistant. The assistant will use the context from your text files to inform its responses.

4. Type `exit` to end the conversation.

### Customization

- Modify `src/prompts.py` to customize the system message and adjust the AI's behavior.
- Update `src/model.py` to use a different Ollama model.
- Adjust chunk size and overlap in `src/document_loader.py` for different document processing behavior.


### Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain) for the awesome framework
- [Ollama](https://ollama.ai/) for local language model execution
- [Llama](https://github.com/meta-llama/llama3) for Meta's open source LLM