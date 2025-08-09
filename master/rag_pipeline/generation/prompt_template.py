multichoice_qa_prompt = """
-- DOCUMENT --
{document}
-- END OF DOCUMENT --

-- INSTRUCTION --
You are a medical expert.
Given the documents, you must answer the question follow these step by step.
First, you must read the question and the options, and draft an answer for it based on your knowledge.
Second, you must read the documents and check if they can help answer the question.
Third, you cross check the document with your knowledge and the draft answer.
Finally, you answer the question based on your knowledge and the true documents.
Your response must end with the letter of the most correct option like: "the answer is A".
The entire thought must under 500 words long.
-- END OF INSTRUCTION --

-- QUESTION --
{question}
{options}
-- END OF QUESTION --
"""

qa_prompt = """
-- DOCUMENT --
{document}
-- END OF DOCUMENT --

-- INSTRUCTION --
You are a medical expert.
Given the documents, you must answer the question follow these step by step.
First, you must read the question and draft an answer for it based on your knowledge.
Second, you must read the documents and check if they can help answer the question.
Third, you cross check the document with your knowledge and the draft answer.
Finally, you answer the question based on your knowledge and the true documents concisely.
Your response must as shortest as possible, in Vietnamese and between brackets like: "[...]".
-- END OF INSTRUCTION --

-- QUESTION --
{question}
-- END OF QUESTION --
"""

retrieve_chatbot_prompt = """
You are a medical expert.
You are having a conversation with a {role} and you have an external documents to help you.
Continue the conversation based on the chat history, the context information, and not prior knowledge.
Before use the retrieved chunk, you must check if it is relevant to the user query. If it is not relevant, you must ignore it.
You use the relevant chunk to answer the question and cite the source inside <<<>>>.
If you don't know the answer, you must say "I don't know".
---------------------
{documents}
---------------------
Given the documents and not prior knowledge, continue the conversation.
---------------------
{conversation}
---------------------
"""

request_retrieve_prompt = """
--- INSTRUCTION ---
You are having a conversation with a {role}.
You have to provide a short query to retrieve the documents that you need inside the brackets like: "[...]".
If it is something do not related to medical field, or something you do not need the external knowledge to answer, you must write "[NO NEED]".
--- END OF INSTRUCTION ---

--- COVERSATION ---
{conversation}
--- END OF COVERSATION ---
"""

answer_prompt = """
-- INSTRUCTION --
You are a medical expert.
Given the documents below, you must answer the question step by step.
First, you must read the question.
Second, you must read the documents and check for it's reliability.
Third, you cross check with your knowledge.
Finally, you answer the question based on your knowledge and the true documents.

Your answer must UNDER 50 words, write on 1 line and write in Vietnamese.
-- END OF INSTRUCTION --

-- QUESTION --
{question}
-- END OF QUESTION --

-- DOCUMENT --
{document}
-- END OF DOCUMENT --

"""

translate_prompt = """
[ INSTRUCTION ]
You are a Medical translator expert.
Your task is to translate this English question into Vietnamese with EXACTLY the same format and write in 1 line.
[ END OF INSTRUCTION ]

[ QUERY TO TRANSLATE ]
{query}
[ END OF QUERY TO TRANSLATE ]
"""

pdf2txt_prompt = """
Rewrite this plain text from pdf file follow the right reading order and these instructions:
- Use markdown format.
- Use same language.
- Keep the content intact.
- Beautify the table.
- No talk.

[ QUERY ]
{query}
[ END OF QUERY ]
"""