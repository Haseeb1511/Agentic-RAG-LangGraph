from src.agent.model_loader import model

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph import StateGraph,START,END
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.all_nodes.nodes import GraphNodes,AgenticRAG


# this line for google embedding as it require running event loop
# GoogleGenerativeAIEmbeddings internally initializes a gRPC async client.
# Streamlit runs your script in a separate thread (ScriptRunner.scriptThread), where no asyncio loop is set by default.
# So when gRPC tries to grab the current event loop → it crashes with
# RuntimeError: There is no current event loop in thread 'ScriptRunner.scriptThread'.
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nodes = GraphNodes(GoogleGenerativeAIEmbeddings(model="models/embedding-001"))


class GraphBuilder:
    def __init__(self):
        db_path = os.path.abspath("./chat_hist/chat.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(database=db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn=conn)
        self.app = None

    def build_graph(self):
        graph = StateGraph(AgenticRAG)

        graph.add_node("Document_Loader",nodes.Document_Loader,)
        graph.add_node("Text_Splitter",nodes.Text_Splitter)
        graph.add_node("Create_Vector_Store",nodes.Create_Vector_Store)
        graph.add_node("Load_Vector_Store",nodes.Load_Vector_Store)  
        graph.add_node("Retriever",nodes.Retriever)
        graph.add_node("Agent",nodes.Agent)
        #Conditional Edge
        graph.add_conditional_edges(START,nodes.check_pdf_or_not,{"create":"Document_Loader",
                                                            "load":"Load_Vector_Store"})

        # If new Vectorstore
        graph.add_edge("Document_Loader","Text_Splitter")
        graph.add_edge("Text_Splitter","Create_Vector_Store")
        graph.add_edge("Create_Vector_Store","Retriever")

        # if Loading VectorStore
        graph.add_edge("Load_Vector_Store","Retriever")

        graph.add_edge("Retriever", "Agent")
        graph.add_edge("Agent", END)

        self.app = graph.compile(checkpointer=self.checkpointer)
        return self.app
    
    def retrieve_all_thread(self):
        all_thread = set()
        for checkpoint in self.checkpointer.list(None):
            all_thread.add(checkpoint.config["configurable"]["thread_id"])
        return list(all_thread)

    def __call__(self):  # __call__ == It lets an instance of your class be called like a function
        return self.build_graph()