from langchain.prompts import PromptTemplate

def get_chat_prompt():
    # System message to set the behavior of the AI
    system_template = """You are an AI assistant named Claude. Your primary goal is to help users with their questions and tasks.
    {rag_instruction}

    Human: {human_input}
    AI: """

    rag_instruction = "Use the following pieces of context to answer the human's questions. The information you have is about US Laws and Bills. Please provide a clear and concise answer, knowing that you are to provide information regarding US Laws and Bills. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}"
    no_rag_instruction = "Answer the human's questions to the best of your ability based on your training."

    return PromptTemplate(
        input_variables=["context", "human_input"],
        template=system_template.format(rag_instruction=rag_instruction, human_input="{human_input}")
    ), PromptTemplate(
        input_variables=["human_input"],
        template=system_template.format(rag_instruction=no_rag_instruction, human_input="{human_input}")
    )