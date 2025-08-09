from ..rag_pipeline import ChatAssistant
from ..rag_pipeline import request_retrieve_prompt

cb = ChatAssistant("mistral-medium", "mistral")

query = "Beta blocker for hypertension"
query = request_retrieve_prompt.format(conversation=query, role="customer")
response = cb.get_response(user=query)
print(response)