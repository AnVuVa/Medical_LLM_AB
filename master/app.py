import gradio as gr
from datetime import datetime

from .rag_pipeline import ChatAssistant, get_embeddings, vretrieve, retrieve_chatbot_prompt, request_retrieve_prompt
from .utils import load_local


# DEVELOPER: Add or remove models here.
AVAILABLE_MODELS = {
    # "mistral large (mistral)": ("mistral-large-2", "mistral"),
    "mistral medium (mistral)": ("mistral-medium", "mistral"),
    "mistral small (mistral)": ("mistral-small", "mistral"),
    "llama3 8B" : ("llama3:8b", "ollama"),
    "llama3.1 8B": ("llama3.1:8b", "ollama"),
    "gpt-oss 20B": ("gpt-oss-20b", "ollama"),
    "gemma3 12B": ("gemma3:12b", "ollama"),
    "gpt 4o mini": ("gpt-4o-mini", "openai"),
    "gpt 4o": ("gpt-4o", "openai"),
}
DEFAULT_MODEL_KEY = "mistral medium (mistral)"

EMBEDDING_MODEL_ID = "alibaba-nlp/gte-multilingual-base"
VECTORSTORE_PATH = "notebook/An/master/knowledge/vectorstore_full"
LOG_FILE_PATH = "log.txt"
MAX_HISTORY_CONVERSATION = 50

# System prompt for the medical assistant
sys = """
You are an Medical Assistant specialized in providing information and answering questions related to healthcare and medicine.
You must answer professionally and empathetically, taking into account the user's feelings and concerns.
"""

# --- Initial Setup (runs once) ---
print("Initializing models and data...")
embedding_model = get_embeddings(EMBEDDING_MODEL_ID, show_progress=False)
vectorstore, docs = load_local(VECTORSTORE_PATH, embedding_model)
print("Initialization complete.")


# --- Helper Functions ---
def log(log_txt: str):
    """Appends a log entry to the log file."""
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(log_txt + "\n")


# --- Core Chatbot Logic ---
def chatbot_logic(message: str, history: list, selected_model_key: str):
    """
    Handles the main logic for receiving a message, performing RAG, and generating a response.
    """
    # 1. Look up the model_id and model_provider from the selected key
    model_id, model_provider = AVAILABLE_MODELS[selected_model_key]

    log(f"** Current time **: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"** User message **: {message}")
    log(f"** Using Model **: {model_id} ({model_provider})")

    # Initialize the assistant with the specified model for this request
    try:
        chat_assistant = ChatAssistant(model_id, model_provider)
    except Exception as e:
        yield f"Error: Could not initialize the model. Please check the ID and provider. Details: {e}"
        return

    # --- RAG Pipeline ---
    # 2. Format conversation history for context
    history = history[-MAX_HISTORY_CONVERSATION:]
    conversation = "".join(f"User: {user_msg}\nBot: {bot_msg}\n" for user_msg, bot_msg in history)
    query_for_rag = conversation + f"User: {message}\nBot:"

    # 3. Generate a search query from the conversation
    rag_query = chat_assistant.get_response(request_retrieve_prompt.format(role="user", conversation=query_for_rag))
    rag_query = rag_query[rag_query.lower().rfind("[") + 1: rag_query.rfind("]")]

    # 4. Retrieve relevant documents if necessary
    if "NO NEED" not in rag_query:
        retrieve_results = vretrieve(rag_query, vectorstore, docs, k=4, metric="mmr", threshold=0.7)
    else:
        retrieve_results = []

    retrieved_docs = "\n".join([f"Document {i+1}:\n" + doc.page_content for i, doc in enumerate(retrieve_results)])
    log(f"** RAG query **: {rag_query}")
    log(f"** Retrieved documents **:\n{retrieved_docs}")

    # --- Final Response Generation ---
    # 5. Create the final prompt with retrieved context
    final_prompt = retrieve_chatbot_prompt.format(role="user", documents=retrieved_docs, conversation=query_for_rag)

    # 6. Stream the response from the LLM
    response = ""
    for token in chat_assistant.get_streaming_response(final_prompt, sys):
        response += token
        yield response
    
    log(f"** Bot response **: {response}")
    log("=" * 50 + "\n\n")

# --- UI Helper Function ---
def start_new_chat():
    """Clears the chatbot and input box to start a new conversation."""
    return None, ""

# --- Gradio UI ---
with gr.Blocks(theme="soft") as chatbot_ui:
    gr.Markdown("# MedLLM")
    gr.Markdown("Your conversations are automatically saved to `log.txt` for future reference.")
    
    model_selector = gr.Dropdown(
        label="Select Model",
        choices=list(AVAILABLE_MODELS.keys()),
        value=DEFAULT_MODEL_KEY, 
    )
        
    chatbot = gr.Chatbot(label="Chat Window", height=500, bubble_full_width=False, value=None)
    
    with gr.Row():
        new_chat_btn = gr.Button("✨ New Chat")
        msg_input = gr.Textbox(
            label="Your Message", 
            placeholder="Type your question here and press Enter...", 
            scale=7 # Make the textbox take more space in the row
        )

    def respond(message, chat_history, selected_model_key):
        """Wrapper function to connect chatbot_logic with Gradio's state."""
        # If chat_history is None (cleared), initialize it as an empty list
        chat_history = chat_history or []
        bot_message_stream = chatbot_logic(message, chat_history, selected_model_key)
        chat_history.append([message, ""])
        for token in bot_message_stream:
            chat_history[-1][1] = token
            yield chat_history

    # Event handler for submitting a message
    msg_input.submit(
        respond,
        [msg_input, chatbot, model_selector],
        [chatbot]
    ).then(
        lambda: gr.update(value=""), None, [msg_input], queue=False
    )

    # Event handler for the "New Chat" button
    new_chat_btn.click(
        start_new_chat,
        inputs=None,
        outputs=[chatbot, msg_input],
        queue=False # Use queue=False for instantaneous UI updates
    )


# --- Launch the App ---
if __name__ == "__main__":
    chatbot_ui.launch(debug=True, share=True)