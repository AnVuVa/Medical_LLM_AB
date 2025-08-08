import gradio as gr
from .rag_pipeline import ChatAssistant, get_embeddings, vretrieve, rerank
from .utils import load_local
from .rag_pipeline import retrieve_chatbot_prompt, request_retrieve_prompt


cb = ChatAssistant("mistral-large-2", "mistral")
sys = """
You are talking to a customer.
I will give you the conversation you have had with the customer so far.
You will respond to the customer's latest message.
"""

embed_model = get_embeddings("alibaba-nlp/gte-multilingual-base", show_progress=False)
vectorestore, docs = load_local("notebook/An/master/knowledge/vectorstore_full", embed_model)      

def chatbot(input_text, history:str, role: str = "customer"):


    conversation_history = "\n".join([f"User: {user_msg}\nBot: {bot_msg}" for user_msg, bot_msg in history])
    conversation_history += f"\nUser: {input_text}\nBot:"

    rag_query = cb.get_response(request_retrieve_prompt.format(conversation=conversation_history, role=role), sys)
    retrieve_results = []
    if "NO" not in rag_query:
        retrieve_results = vretrieve(rag_query, vectorestore, docs, 5, "cosine", 0.5)
        retrieve_results = rerank(retrieve_results)

    rag_context = retrieve_chatbot_prompt.format(retrieved_chunk=retrieve_results, conversation=conversation_history, role=role)
    response = cb.get_response(conversation_history, sys)
    history.append((input_text, response))

    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"User: {input_text}\nBot: {response}\n")
        f.write(f"RAG Query: {rag_query}\n")
        f.write(f"RAG Context: {rag_context}\n")
        f.write(f"Response: {response}\n\n\n")

    return history, history

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>ChatGPT</h1>")
    gr.Markdown("<p style='text-align: center;'>A simple chatbot</p>")

    chatbot_interface = gr.Chatbot()
    input_text = gr.Textbox(label="Your Message", placeholder="Type your message here...")

    def submit(input_text, history):
        return chatbot(input_text, history)

    input_text.submit(submit, [input_text, chatbot_interface], [chatbot_interface, chatbot_interface])

    gr.Markdown("<p style='text-align: center;'>Powered by Gradio</p>")

demo.launch(share=True)