from model import get_ollama_model
from document_loader import load_documents
from rag import setup_rag
from prompts import get_chat_prompt

def main():
    # Load the Ollama model
    model = get_ollama_model()

    print("Model loaded successfully.")

    # Load documents from the context folder
    documents = load_documents()

    print("Documents loaded successfully.")

    # Set up RAG
    qa_chain = setup_rag(model, documents)

    print("RAG setup successfully.")

    # Get the chat prompts
    rag_prompt, no_rag_prompt = get_chat_prompt()

    print("Prompts loaded successfully.")

    print("Welcome to the AI assistant. Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Prepare the full prompt
        if qa_chain:
            full_prompt = rag_prompt.format(context="", human_input=user_input)
        else:
            full_prompt = no_rag_prompt.format(human_input=user_input)

        # Get the response from the model
        if qa_chain:
            result = qa_chain({"query": full_prompt})
            print("AI:", result['result'])
        else:
            # If RAG is not available, use the model directly
            result = model.invoke(full_prompt)
            print("AI:", result)

if __name__ == "__main__":
    main()