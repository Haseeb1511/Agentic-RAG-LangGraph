
# My Checklist

- [x] Agentic vector store
- [x] memory
- [ ] streaming  
- [ ] nodes OOP
- [ ] model OOP(Config)  
- [ ] Docker
- [ ] Fast API backend




Full persistence: all messages stored in SQLite via LangGraph’s checkpointer.

Cross-session memory: even if you restart the server, previous conversations are still accessible.

Scales to multiple threads and users.

Good for production-level RAG bots where persistent conversation history is needed.


SQLite checkpointer = your diary → stores all past conversations.

Including past messages in the prompt = reading your diary before answering → now the AI can actually “remember” what happened.


```bash
Great question! Here's the scoop on memory usage in LangChain and LangGraph as of August 2025:

---

## Official Recommendation: Use LangGraph Persistence

* LangChain has **deprecated legacy memory classes** such as `ConversationBufferMemory`, `ConversationBufferWindowMemory`, `ConversationSummaryMemory`, and `ConversationEntityMemory`. Instead, the **recommended approach** is to leverage **LangGraph’s persistence system** using checkpointers like `SqliteSaver` or `InMemorySaver` to manage memory. ([Focused][1])

* LangGraph treats memory as **state persisted via the graph**, not just message history—it can include uploaded files, retrieved documents, and any other data the graph manages. Crucially, this state is checkpointed and **thread-scoped**, allowing conversation threads to be resumed later. ([langchain-ai.github.io][2], [Medium][3])

---

## Memory Types in LangGraph

LangGraph supports two main forms of memory:

1. **Short-Term Memory**

   * Chat history and related thread-specific state stored persistently via checkpointers. This enables you to resume a conversation by loading the latest state of a thread. ([langchain-ai.github.io][2])

2. **Long-Term Memory (Cross-Thread)**

   * Stores information across conversations using namespaces and document stores. This lets agents recall facts, user preferences, and histories beyond a single chat thread. ([LangChain Blog][4], [langchain-ai.github.io][2])

LangGraph exposes memory types like semantic, episodic, and procedural, but at its core it’s all persisted in the graph state—no separate memory API is needed. ([langchain-ai.github.io][2])

---

## Perspective from the LangGraph Team

On the official LangGraph discussions board, a maintainer clarified:

> *“In LangGraph, memory is just checkpointing/persistence … All you have to do is compile it with a checkpointer, then invoke the graph with a thread\_id … your full graph has memory.”* ([GitHub][5])

This reinforces the official stance: **LangGraph’s built-in persistence system is the preferred memory mechanism.**

---

## At a Glance

| Framework                | Memory Approach                                             | Notes                                                                                                                        |
| ------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **LangChain Legacy**     | ConversationBufferMemory, SummaryMemory, EntityMemory, etc. | **Deprecated by August 2025** ([Focused][1])                                                                                 |
| **LangGraph (Official)** | Persistence via graph state & checkpointers                 | **Recommended**; supports thread memory and cross-thread long-term memory ([langchain-ai.github.io][2], [LangChain Blog][4]) |

---

### Summary

As of August 2025, the **most used and officially recommended approach** in the LangChain ecosystem is to rely on **LangGraph memory via checkpointer persistence**. This method replaces legacy memory classes and offers robust capabilities for both short-term, thread-specific memory and long-term, cross-session memory.

Let me know if you'd like help migrating your existing memory setup to LangGraph or setting up long-term memory with namespaces!

[1]: https://focused.io/lab/customizing-memory-in-langgraph-agents-for-better-conversations?utm_source=chatgpt.com "Customizing Memory in LangGraph Agents for Better ..."
[2]: https://langchain-ai.github.io/langgraph/concepts/memory/?utm_source=chatgpt.com "LangGraph memory - Overview"
[3]: https://medium.com/%40areebaayub2908/using-langgraph-memory-for-persistent-chat-conversations-f44358dd21f5?utm_source=chatgpt.com "Using LangGraph Memory for Persistent Chat Conversations"
[4]: https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/?utm_source=chatgpt.com "Launching Long-Term Memory Support in LangGraph"
[5]: https://github.com/langchain-ai/langgraph/discussions/352?utm_source=chatgpt.com "What's the proper way to use memory with langgraph? #352"


```