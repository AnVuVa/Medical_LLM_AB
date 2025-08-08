import gradio as gr

from .rag_pipeline import ChatAssistant, get_embeddings, vretrieve, retrieve_chatbot_prompt, request_retrieve_prompt
from .utils import load_local

CHAT_MODEL_ID = "mistral-medium"
CHAT_MODEL_PROVIDER = "mistral"
EMBEDDING_MODEL_ID = "alibaba-nlp/gte-multilingual-base"
VECTORSTORE_PATH = "notebook/An/master/knowledge/vectorstore_full"
LOG_FILE_PATH = "log.txt"
MAX_HISTORY_CONVERSATION = 50

sys = """
You are an Medical Assistant specialized in providing information and answering questions related to healthcare and medicine.
You must answer professionally and empathetically, taking into account the user's feelings and concerns.
"""

print("Initializing models and data...")
chat_assistant = ChatAssistant(CHAT_MODEL_ID, CHAT_MODEL_PROVIDER)
embedding_model = get_embeddings(EMBEDDING_MODEL_ID, show_progress=False)
vectorstore, docs = load_local(VECTORSTORE_PATH, embedding_model)
print("Initialization complete.")

def log(log_txt:str):
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(log_txt + "\n")

def process_query(query: str) -> str:
    rag_query = chat_assistant.get_response(request_retrieve_prompt.format(role="user", conversation=query))
    rag_query = rag_query[rag_query.lower().rfind("["): rag_query.rfind("]")+1]

    if "NO" not in rag_query:
        retrieve_results = vretrieve(rag_query, vectorstore, docs, k=4, metric="mmr", threshold=0.7)
    else:
        retrieve_results = []

    retrieved_docs = "\n".join([f"Document {i+1}:\n" + doc.page_content for i, doc in enumerate(retrieve_results)])
    log(f"Retrieved documents:\n{retrieved_docs}")
    log(f"RAG query: {rag_query}")
    return retrieve_chatbot_prompt.format(role="user", documents=retrieved_docs, conversation=query)

def process(message: str, history: list[list[str]]) -> str:
    log(f"User message: {message}")
    history = history[-MAX_HISTORY_CONVERSATION:]
    conversation = "".join(f"User: {history[i][0]}\nBot: {history[i][1]}\n" for i in range(len(history)))
    query = conversation + f"User: {message}\nBot:"
    query = process_query(query)
    return chat_assistant.get_response(query, sys)

# --- Chatbot Logic ---
def chatbot_logic(message: str, history: list) -> any:
    response = process(message, history)
    log(f"Bot response: {response}")
    log("="*50 + "\n\n")
    yield response

# Create the Gradio interface
chatbot_ui = gr.ChatInterface(
    fn=chatbot_logic,
    title="MedLLM",
    theme="soft",
)

# Launch the app
if __name__ == "__main__":
    chatbot_ui.launch(debug=True, share=True)
