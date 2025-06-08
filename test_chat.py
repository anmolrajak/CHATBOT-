# test_chat.py
from rag.chat import KnowledgeChatbot

chatbot = KnowledgeChatbot(faiss_index_path="data/faiss_index")
query = "How many r in strawberry?"

response = chatbot.ask(query)
print("\n🤖 Answer:\n", response)
