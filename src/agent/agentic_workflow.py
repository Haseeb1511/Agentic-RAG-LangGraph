
from src.all_nodes.nodes import Document_Loader,Text_Splitter,Create_Vector_Store,Load_Vector_Store,Retriever,Agent,check_pdf_or_not,AgenticRAG
from src.agent.model_loader import model

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph import StateGraph,START,END
import os

class GraphBuilder:
    def __init__(self):
        pass

    def build_graph(self):
        db_path = os.path.abspath("./chat_hist/chat.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(database=db_path, check_same_thread=False)

        graph = StateGraph(AgenticRAG)
        checkpointer = SqliteSaver(conn=conn)

        graph.add_node("Document_Loader",Document_Loader,)
        graph.add_node("Text_Splitter",Text_Splitter)
        graph.add_node("Create_Vector_Store",Create_Vector_Store)
        graph.add_node("Load_Vector_Store",Load_Vector_Store)  
        graph.add_node("Retriever",Retriever)
        graph.add_node("Agent",Agent)
        #Conditional Edge
        graph.add_conditional_edges(START,check_pdf_or_not,{"create":"Document_Loader",
                                                            "load":"Load_Vector_Store"})

        # If new Vectorstore
        graph.add_edge("Document_Loader","Text_Splitter")
        graph.add_edge("Text_Splitter","Create_Vector_Store")
        graph.add_edge("Create_Vector_Store","Retriever")

        # if Loading VectorStore
        graph.add_edge("Load_Vector_Store","Retriever")

        graph.add_edge("Retriever", "Agent")
        graph.add_edge("Agent", END)

        app = graph.compile(checkpointer=checkpointer)
        return app

    def __call__(self):  # __call__ == It lets an instance of your class be called like a function
        return self.build_graph()