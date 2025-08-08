from ..rag_pipeline import ChatAssistant

cb = ChatAssistant("llama3:8b", "ollama")
response = cb.get_response("What is the capital of France?")
print(response)