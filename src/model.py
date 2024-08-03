from langchain_community.llms import Ollama

def get_ollama_model(model_name="llama3.1"):
    return Ollama(model=model_name)