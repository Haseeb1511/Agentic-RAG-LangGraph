#hreads in this project  Uploaded pdf,Legal🏛️Psychiatrist🧠,Dermatology🩺


from fastapi import FastAPI,HTTPException,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os,sys
from langchain_core.messages import HumanMessage,AIMessage
from src.agent.agentic_workflow import GraphBuilder

import shutil  #It is mainly used for copying, moving, archiving, and deleting files or directories.  better than OS module
from pathlib import Path

# Add project root to sys.path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = FastAPI(title="Agentic RAG Backend")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query:str
    thread_id:str
    vectorstore_path: Optional[str] = None
    documents_path: Optional[str] = None

class QueryResponse(BaseModel):
    answer:str
    thread_id:str

class ChatMessage(BaseModel):
    role:str
    content:str

class ChatHistoryResponse(BaseModel):
    messages : list[ChatMessage]


@app.get("/")
async def root():
    return {"message":"ai agent is running"}

graph = GraphBuilder()
workflow = graph.build_graph()


uploaded_pdfs_store = {}

VECTORSTORE_PATHS = {
    "Dermatology🩺": "./vectorstores/dermatology_faiss",
    "Psychiatrist🧠": "./vectorstores/psychiatrist_faiss",
    "Legal🏛️": "./vectorstores/legal_faiss"
}


#step 1: load conversation from sqlite.db
#step2 :  handle upload pdf

def load_conversation(thread_id:str):
    state = workflow.get_state(config={"configurable":{"thread_id":thread_id}})
    messages = state.values.get("messages",[])
    chat_messages = []
    for message in messages:
        if isinstance(message,HumanMessage):
            role = "user"
        elif isinstance(message,AIMessage):
            role="assistant"
        else:
            role="system"
        chat_messages.append(ChatMessage(role=role,content=message.content))
    return chat_messages
    


# UploadFile has these attributes:
# filename: original file name from the client.  --->uploade_file.name
# content_type: MIME type (e.g., "application/pdf").---->application/pdf
# file: an internal SpooledTemporaryFile (file-like object you can .read(), .write(), .seek()).---->upload_file.getvalue()
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile): # #upload file is built in fast api validator
    """This function save the uploaded pdf file to temp folder
    and inside dict like {"pdf_path":"path to pdf"
                            "vectorstore_path":"path to vectorstore"}
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400,detail="only pdf are allowed")
    
    #create temp directory for uploaded PDF
    temp_dir = Path("./temp_pdfs")
    temp_dir.mkdir(exist_ok=True)

    #save uploaded file
    filename_no_ext = Path(file.filename).stem #Returns only the filename without its extension.---?file_name no extension
    pdf_choice = f"PDF_{filename_no_ext}"
    temp_pdf_path = temp_dir / f"PDF_{filename_no_ext}.pdf"

    with open (temp_pdf_path,"wb") as f:
        shutil.copyfileobj(file.file,f) #copy data chunk by chunk in to temp_pdf_path

    # Store in our tracking dictionary       [[we only store PDF path no vectorstore path is needed here]]
    uploaded_pdfs_store[pdf_choice] = str(temp_pdf_path)

    return {
        "message":"Pdf uploaded successfully",
        "pdf_choice":pdf_choice,
        "filename":file.filename
    }


@app.get("/vectorstores")
async def get_vectorstores():
    "Get List of Avaliable Vectorstore"
    base_choices = ["Dermatology🩺", "Psychiatrist🧠", "Legal🏛️"]
    uploaded_pdfs = list(uploaded_pdfs_store.keys()) # we get name of uploaded pdf and add it into base chocie
    return{
        "base_choices":base_choices,
        "uploaded_pdfs":uploaded_pdfs,
        "all_choices":base_choices+uploaded_pdfs
    }



@app.post("/query",response_model=QueryResponse)
async def process_query(request:QueryRequest):
    """This function let user chat with PDF + Vectorstores"""
    thread_id = request.thread_id   # first we get thread(choice) which we want to send our query to
    CONFIG = {"configurable":{"thread_id":thread_id}}

    # Handling PDF thread
    if thread_id.startswith("PDF_"):
        filename_no_ext = thread_id.replace("PDF_", "")
        temp_pdf_path = uploaded_pdfs_store[thread_id]  # we will use thread id to access that specific pdf path from above [uploaded_pdf_store]
        
        input_data = {
        "documents_path": temp_pdf_path,
        "vectorstore_path": f"./Vectorstores/{filename_no_ext}/",
        "query": request.query}
                    
    else:
        input_data = {
            "vectorstore_path":VECTORSTORE_PATHS[thread_id],
            "query":request.query
        }

    result = workflow.invoke(input_data,config=CONFIG)

    if not result or "answer" not in result:
        raise HTTPException(status_code=500, detail="Failed to generate response")
    return QueryResponse(
        answer=result["answer"],
        thread_id=thread_id
    )
        

# load past history from DB
@app.get("/chat_history/{thread_id}",response_model=ChatHistoryResponse)
async def get_chat_history(thread_id:str):
    """Get chat history for a specific chat from Database"""
    messages = load_conversation(thread_id=thread_id)
    return ChatHistoryResponse (messages = messages)


app.get("/graph-visualization")
async def get_graph_visualization():
    """Generate and return graph visualization"""
    graph_png = workflow.get_graph().draw_mermaid_png()
        
    # Save graph visualization
    graph_path = Path("graph.png")
    with open(graph_path, "wb") as f:
        f.write(graph_png)
        
    return {"message": "Graph visualization saved as graph.png"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

# uvicorn <module_name>:<app_instance> [options]
#uvicorn backend.app:app --reload
