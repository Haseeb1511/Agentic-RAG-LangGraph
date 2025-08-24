from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_community.vectorstores import FAISS

import os
from src.agent.model_loader import model
from src.prompt_library.prompt import prompt_template

from typing import TypedDict,Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages


# from dotenv import load_dotenv
# load_dotenv()


class AgenticRAG(TypedDict):
    query:str
    documents_path:str
    documents:list[Document]
    chunks:list[Document]
    vectorstore:object
    retrieved_docs:list[Document]
    answer:str
    vectorstore_path:str
    messages: Annotated[list[BaseMessage], add_messages]


class GraphNodes:
    def __init__(self,embeddeing_model):
        self.embeddeing_model= embeddeing_model 

    def Document_Loader(self,state: AgenticRAG):
        path = os.path.abspath(state["documents_path"])  # ensure absolute
        if os.path.isfile(path):  # single PDF case
            loader = PyPDFLoader(path)
            loaded_pdf = loader.load()
        elif os.path.isdir(path):  # directory case
            loader = DirectoryLoader(
                path=path,
                glob="*.pdf",
                loader_cls=PyPDFLoader)
            loaded_pdf = loader.load()
        else:
            raise ValueError(f"Invalid documents_path: {path}")
        return {"documents": loaded_pdf}


    def Text_Splitter(self,state:AgenticRAG):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=199)
        chunks = splitter.split_documents(state["documents"])
        return {"chunks":chunks}


    def Create_Vector_Store(self,state:AgenticRAG):
        embedder = self.embeddeing_model
        vector_store = FAISS.from_documents(documents=state["chunks"],embedding=embedder)
        vector_store.save_local(state["vectorstore_path"])
        return {"vectorstore_path":state["vectorstore_path"]}



    def Load_Vector_Store(self,state:AgenticRAG):
        embedder = self.embeddeing_model
        vector_store = FAISS.load_local(folder_path=state["vectorstore_path"],
                                        embeddings=embedder,
                                        allow_dangerous_deserialization=True)
        return {"vectorstore_path":state["vectorstore_path"]}


    def Retriever(self,state: AgenticRAG):
        vector_store = FAISS.load_local(
            folder_path=state["vectorstore_path"],
            embeddings=self.embeddeing_model,
            allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        query = state["query"]

        docs = retriever.invoke(query)
        return {"retrieved_docs": docs}


    def Agent(self,state: AgenticRAG):
        docs = state["retrieved_docs"]
        context = "\n\n".join([doc.page_content for doc in docs])

        query = state["query"]

        formatted_prompt = prompt_template.format(
            context=context,
            question=query)
        response = model.invoke(formatted_prompt)

        # Save to state (sqlite)
        state.setdefault("messages", [])
        state["messages"].append(HumanMessage(content=state["query"]))
        state["messages"].append(AIMessage(content=response.content))
        return {
            "answer": response.content,
            "messages": state["messages"]
            }

    def check_pdf_or_not(self,state: AgenticRAG):
        if state.get("documents_path") and not os.path.exists(state["vectorstore_path"]):
            return "create"
        else:
            return "load"