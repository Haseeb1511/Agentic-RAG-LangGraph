from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

from src.agent.model_loader import model
from src.prompt_library.prompt import prompt_template
from langchain_core.documents import Document
from typing import TypedDict



# this line for google embedding as it require running event loop
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from dotenv import load_dotenv
load_dotenv()

EMBEDDER = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class AgenticRAG(TypedDict):
    query:str
    documents_path:str
    documents:list[Document]
    chunks:list[Document]
    vectorstore:object
    retrieved_docs:list[Document]
    answer:str
    vectorstore_path:str





def Document_Loader(state:AgenticRAG):
    loader = DirectoryLoader(
        path=state["documents_path"],
        glob="*.pdf",
        loader_cls=PyPDFLoader)
    loaded_pdf = loader.load()
    return {"documents":loaded_pdf}

def Text_Splitter(state:AgenticRAG):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=199)
    chunks = splitter.split_documents(state["documents"])
    return {"chunks":chunks}

def Create_Vector_Store(state:AgenticRAG):
    embedder = EMBEDDER
    vector_store = FAISS.from_documents(documents=state["chunks"],embedding=embedder)
    vector_store.save_local(state["vectorstore_path"])

    return {"vectorstore_path":state["vectorstore_path"]}


def Load_Vector_Store(state:AgenticRAG):
    embedder = EMBEDDER
    vector_store = FAISS.load_local(folder_path=state["vectorstore_path"],
                                    embeddings=embedder,
                                    allow_dangerous_deserialization=True)
    return {"vectorstore_path":state["vectorstore_path"]}


def Retriever(state: AgenticRAG):
    embedder = EMBEDDER
    vector_store = FAISS.load_local(
        folder_path=state["vectorstore_path"],
        embeddings=embedder,
        allow_dangerous_deserialization=True)
    
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(state["query"])
    return {"retrieved_docs": docs}


def Agent(state:AgenticRAG):
    docs = state["retrieved_docs"]
    context = "\n\n".join([doc.page_content for doc in docs])
    formated_prompt = prompt_template.format(
        context = context,
        question = state["query"])
    response = model.invoke(formated_prompt)    
    return{"answer":response.content}


def check_pdf_or_not(state: AgenticRAG):
    if state.get("documents_path") and not os.path.exists(state["vectorstore_path"]):
        return "create"
    else:
        return "load"