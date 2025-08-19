
import sys,os
#Streamlit doesn’t always preserve the root in sys.path  so we add this in order to imort locall modules
# add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.agent.agentic_workflow import GraphBuilder
from pathlib import Path


#==========================================================Title==================================================================================
st.title("Agentic RAG Q&A")



CONFIG = {"configurable":{"thread_id":"1"}}
#=========================================================Session States==========================================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []



#Display messages in session state in Message History
for messages in st.session_state["message_history"]:
    with st.chat_message(messages["role"]):
        st.markdown(messages["content"])

#================================================================MAIN============================================================================

user_input = st.chat_input("Ask Anything....")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
        st.session_state["message_history"].append({"role":"user","content":user_input})

# Build Graph
graph = GraphBuilder()
app = graph.build_graph()

# save graph as PNG
graph_png = app.get_graph().draw_mermaid_png()
with open("graph.png","wb") as f:
    f.write(graph_png)

#=============================================================(CASE:1)Uploaded PDF================================================================

uploaded_pdf = st.sidebar.file_uploader("Upload PDF here to Chat with it")
choice = st.sidebar.radio("Chose From Here:",["Dermatology","psychiatrist","Legal"])
result = None
if user_input:
    if uploaded_pdf:
        # IF FILE IS UPLOADED BY USER
        filename_no_ext = Path(uploaded_pdf.name).stem #get file name without extension
        input_data = {
            "documents_path":uploaded_pdf ,  # document path pdf
            "vectorstore_path": f"../Vectorstores/{filename_no_ext}/",  # provide a path to store vector store
            "query": user_input }  #query
        result = app.invoke(input_data,config=CONFIG)
#==========================================================(CASE:2)Loading Vector Store===========================================================
    else:
        if choice == "Dermatology":
            input_data = {
            "vectorstore_path": "./vectorstores/dermatology_faiss",  # provide a path to store vector store
            "query": user_input } #query
            result = app.invoke(input_data,config=CONFIG)

        elif choice == "psychiatrist":
            input_data = {
            "vectorstore_path": "./vectorstores/psychiatrist_faiss",
            "query": user_input }
            result = app.invoke(input_data,config=CONFIG)

        elif choice == "Legal":
            input_data = {
            "vectorstore_path": "./vectorstores/legal_faiss", 
            "query": user_input }
            result = app.invoke(input_data,config=CONFIG)
    
if result:    
    ai_respone = result["answer"]
    with st.chat_message("ai"):
        st.markdown(ai_respone)
        st.session_state["message_history"].append({"role":"ai","content":ai_respone})