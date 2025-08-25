
# My Checklist

- [x] Agentic vector store
- [x] memory
- [x] nodes OOP
- [x] model OOP(Config)  
- [x] Docker
- [x] Fast API backend
- [ ] streaming  
- [ ] add feature so user can chosee the model to work with
- [ ] add logger and exception hanlding






SQLite checkpointer = your diary → stores all past conversations.



```markdown
## 📂 Project Structure

```

.
├── backend/
│   └── app.py                  # FastAPI backend entrypoint
│
├── frontend/
│   └── streamlit/              # Streamlit frontend app
│
├── src/
│   ├── agent/                  # Agent workflow with LangGraph
│   ├── exception/              # Custom exception handling
│   ├── logger/                 # Logging utility
│   ├── prompt\_library/         # Prompt templates
│   └── all\_node/               # All LangGraph nodes
│
├── notebook/
│   ├── notebook\_with\_conversationbuffermemory.ipynb
│   ├── notebook\_with\_createvectorstore.ipynb
│   ├── notebook\_with\_load\_vectorstore.ipynb
│   └── notebook\_with\_streaming.ipynb
│
├── requirements.txt            # Python dependencies
├── graph.png                   # LangGraph workflow diagram
└── README.md                   # Project documentation

```
