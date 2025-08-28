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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever,ContextualCompressionRetriever
from langchain_community.document_transformers import LongContextReorder,EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.memory import ConversationSummaryMemory
from src.agent.model_loader import summary_llm



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


# Create conversation summary memory
chat_history = InMemoryChatMessageHistory()

memory = ConversationSummaryMemory(
    memory_key="chat_history",
    chat_memory=chat_history,
    llm=summary_llm,
    return_messages=True)


class GraphNodes:
    def __init__(self,embedding_model,reranker_model,summary_llm):
        self.embedding_model= embedding_model 
        self.reranker_model = reranker_model
        self.summary_llm = summary_llm

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
        embedder = self.embedding_model
        vector_store = FAISS.from_documents(documents=state["chunks"],embedding=embedder)
        vector_store.save_local(state["vectorstore_path"])
        return {"vectorstore_path":state["vectorstore_path"]}



    def Load_Vector_Store(self,state:AgenticRAG):
        embedder = self.embedding_model
        vector_store = FAISS.load_local(folder_path=state["vectorstore_path"],
                                        embeddings=embedder,
                                        allow_dangerous_deserialization=True)
        return {"vectorstore_path":state["vectorstore_path"]}


    def Retriever(self,state: AgenticRAG):
        vector_store = FAISS.load_local(
            folder_path=state["vectorstore_path"],
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True)
        
        # Dense retriever
        retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})

        query = state["query"]

        if "chunks" in state:
            ## Sparse retriever
            bm25_retriever = BM25Retriever.from_documents(state["chunks"])
            #if query is short like medical term we take bigger weight of bm25 retriever and if query is long like natual language then we reduce weightage of bm25 retriever
            len_query = len(query.split())
            if len_query<6:
                weights = [0.6,0.4]
            else:
                weights = [0.85, 0.15]  # rely on FAISS more
            
            ensemble_retriever  = EnsembleRetriever(
                retrievers=[retriever,bm25_retriever],
                weights=weights)
        else:
            ensemble_retriever = retriever
        
        # Compression pipeline (rerank + deduplicate + reorder)
        reranker = self.reranker_model # anks documents by how well they answer the user's question.
        filter = EmbeddingsRedundantFilter(embeddings=self.embedding_model) # Removes duplicate or highly similar chunks.
        reordering = LongContextReorder()  # Reorders documents to maximize coherence in long context windows
        pipeline = DocumentCompressorPipeline(transformers=[reranker,filter,reordering])

        compression_retriever = ContextualCompressionRetriever(
            base_compressor= pipeline,
            base_retriever=ensemble_retriever)
    
        docs = compression_retriever.invoke(query)
        return {"retrieved_docs": docs}

    
    def rebuild_memory_from_state(self,state):
        """This Methood Rebuilds ConversationSummaryMemory by replaying saved Human/AI messages from persisted state, making the memory continuous across restarts.
        Run this before Agent function
        AFTER SERVER IS RESTART EMORY WILL STILL BE THIER
        """
        memory.chat_memory.clear()  # clear any old messages
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                memory.chat_memory.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                memory.chat_memory.add_ai_message(msg.content)



    def Agent(self,state: AgenticRAG):
        # 🔑 Always rebuild memory from state before use
        self.rebuild_memory_from_state(state)
        
        docs = state["retrieved_docs"]
        # context = "\n\n".join([doc.page_content for doc in docs])
        context = "\n\n".join([f"Source: {doc.metadata.get('filename', '')}, Page: {doc.metadata.get('page', '')}\n{doc.page_content}"
        for doc in docs])

        # Load summarized history
        past_dialogue = memory.load_memory_variables({})["chat_history"]

        # Format prompt
        formated_prompt = prompt_template.format(
            history=past_dialogue,
            context=context,
            question=state["query"])

        response = model.invoke(formated_prompt)

        # Update summary memory properly
        memory.save_context(
            {"input": state["query"]}, 
            {"output": response.content})

        # Save to state instead of external memory
        state.setdefault("messages", [])
        state["messages"].append(HumanMessage(content=state["query"]))
        state["messages"].append(AIMessage(content=response.content))
        
        return {
            "answer": response.content,
            "messages": state["messages"]}


    def check_pdf_or_not(self,state: AgenticRAG):
        if state.get("documents_path") and not os.path.exists(state["vectorstore_path"]):
            return "create"
        else:
            return "load"