
# My Checklist

- [x] Agentic vector store
- [x] memory
- [ ] streaming  


Full persistence: all messages stored in SQLite via LangGraph’s checkpointer.

Cross-session memory: even if you restart the server, previous conversations are still accessible.

Scales to multiple threads and users.

Good for production-level RAG bots where persistent conversation history is needed.


SQLite checkpointer = your diary → stores all past conversations.

Including past messages in the prompt = reading your diary before answering → now the AI can actually “remember” what happened.