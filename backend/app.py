#hreads in this project  Uploaded pdf,Legal🏛️Psychiatrist🧠,Dermatology🩺


from fastapi import FastAPI,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from typing import List,Optional
from fastapi.responses import JSONResponse
import os,sys
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage,SystemMessage
from src.agent.agentic_workflow import GraphBuilder

import shutil
import tempfile
from pathlib import Path

# Add project root to sys.path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = FastAPI(title="Agentic RAG Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # ← Allow ANY origin (public)
    allow_credentials=True,      # ← Allow cookies, auth headers
    allow_methods=["*"],         # ← Allow ALL HTTP methods (GET, POST, PUT, etc.)
    allow_headers=["*"],         # ← Allow ALL custom headers
)


# ================================================================ Models ======================================================================

class QueryRequest(BaseModel):
    query: str
    thread_id: str
    vectorstore_path: Optional[str] = None
    documents_path: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    thread_id: str

class ChatMessage(BaseModel):
    """Message to be displayed in streamlit Chatmessage"""
    role: str
    content: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]

#============================================== Health Check ===========================================================================
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Agentic RAG Q&A API is running!"}


#===================================================== Building Graph ========================================================================
graph = GraphBuilder()
workflow = graph.build_graph()


# =========================================================== Predefined Vectorstore Paths =======================================================
# Store uploaded PDFs temporarily
uploaded_pdfs_store = {}

VECTORSTORE_PATHS = {
    "Dermatology🩺": "./vectorstores/dermatology_faiss",
    "Psychiatrist🧠": "./vectorstores/psychiatrist_faiss",
    "Legal🏛️": "./vectorstores/legal_faiss"
}

 # ==================================================== Load Conversation ==================================================================

def load_conversation(thread_id: str):
    """Load conversation history for a specific thread"""
    try:
        state = workflow.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        
        chat_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "system"
            
            chat_messages.append(ChatMessage(role=role, content=message.content))
        return chat_messages
    except Exception as e:
        return []


    

#=================================================== Uploaded PDF ========================================================= 


# UploadFile has these attributes:
# filename: original file name from the client.  --->uploade_file.name
# content_type: MIME type (e.g., "application/pdf").---->application/pdf
# file: an internal SpooledTemporaryFile (file-like object you can .read(), .write(), .seek()).---->upload_file.getvalue()
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile): # #upload file is built in fast api validator
    """Upload and store PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("./temp_pdfs")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        filename_no_ext = Path(file.filename).stem
        pdf_choice = f"PDF_{filename_no_ext}"
        temp_pdf_path = temp_dir / f"{filename_no_ext}.pdf"
        
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)  ##copies data in chunks and write to 
        
        # Store in our tracking dictionary
        uploaded_pdfs_store[pdf_choice] = str(temp_pdf_path)
        
        return {
            "message": "PDF uploaded successfully",
            "pdf_choice": pdf_choice,
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


#=================================================== Avaliable Veector stores ========================================================= 
@app.get("/vectorstores")
async def get_vectorstores():
    """Get list of available vectorstores"""
    base_choices = ["Dermatology🩺", "Psychiatrist🧠", "Legal🏛️"]
    uploaded_pdfs = list(uploaded_pdfs_store.keys())
    return {
        "base_choices": base_choices,
        "uploaded_pdfs": uploaded_pdfs,
        "all_choices": base_choices + uploaded_pdfs
    }


#============================================================ Query =====================================================================

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
        thread_id = request.thread_id
        CONFIG = {"configurable": {"thread_id": thread_id}}
        # Handle PDF threads
        if thread_id.startswith("PDF_"):
            filename_no_ext = thread_id.replace("PDF_", "")
            temp_pdf_path = uploaded_pdfs_store[thread_id]
            
            input_data = {
                "documents_path": temp_pdf_path,
                "vectorstore_path": f"../Vectorstores/{filename_no_ext}/",
                "query": request.query
            }
        # Handle prebuilt vectorstores
        else:
            if thread_id not in VECTORSTORE_PATHS:
                raise HTTPException(status_code=404, detail="Vectorstore not found")
            
            input_data = {
                "vectorstore_path": VECTORSTORE_PATHS[thread_id],
                "query": request.query
            }
        
        # Process query through the graph
        result = workflow.invoke(input_data, config=CONFIG)
        
        if not result or "answer" not in result:
            raise HTTPException(status_code=500, detail="Failed to generate response")
        
        return QueryResponse(
            answer=result["answer"],
            thread_id=thread_id
        )


@app.get("/chat-history/{thread_id}", response_model=ChatHistoryResponse)
async def get_chat_history(thread_id: str):
    """Get chat history for a specific thread"""
    messages = load_conversation(thread_id)
    return ChatHistoryResponse(messages=messages)
   

@app.get("/all-threads")
async def get_all_threads():
    """Get all available chat threads"""
    all_threads = workflow.retrieve_all_thread()
    return {"threads": all_threads}



app.get("/graph-visualization")
async def get_graph_visualization():
    """Generate and return graph visualization"""
    graph_png = workflow.get_graph().draw_mermaid_png()
        
    # Save graph visualization
    graph_path = Path("graph.png")
    with open(graph_path, "wb") as f:
        f.write(graph_png)
        
    return {"message": "Graph visualization saved as graph.png"}




# if __name__ == "__main__":
#     # import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)








# uvicorn <module_name>:<app_instance> [options]
#uvicorn backend.app:app --reload
