import sys,os
# Add project root to sys.path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.agent.agentic_workflow import GraphBuilder
from pathlib import Path

# ================================================================ Build Graph Once ==============================================================
if "graph_builder" not in st.session_state:
    st.session_state.graph_builder = GraphBuilder()
if "graph_app" not in st.session_state:
    st.session_state.graph_app = st.session_state.graph_builder.build_graph()

graph = st.session_state.graph_builder
app = st.session_state.graph_app

# =========================================================== Page Title =========================================================================
st.title("Agentic RAG Q&A")

# ======================================================= User Input =============================================================================
user_input = st.chat_input("Ask Anything....")

# choice list
base_choices = ["Dermatology", "Psychiatrist", "Legal"]

# Keep track of uploaded PDFs
if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = []

uploaded_pdf = st.sidebar.file_uploader("Upload PDF here to Chat with it")

# Add uploaded PDF dynamically to choices
if uploaded_pdf:
    filename_no_ext = Path(uploaded_pdf.name).stem
    pdf_choice = f"PDF_{filename_no_ext}"

    if pdf_choice not in st.session_state["uploaded_pdfs"]:
        # Save the file temporarily so it can be reused
        temp_pdf_path = f"./temp_{uploaded_pdf.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.session_state["uploaded_pdfs"].append(pdf_choice)

choices = base_choices + st.session_state["uploaded_pdfs"]

# Sidebar radio includes both prebuilt and PDFs
choice = st.sidebar.radio("Choose From Here:", choices)

# ======================================================= Thread ID & Config =====================================================================
thread_id = choice
CONFIG = {"configurable": {"thread_id": thread_id}}

# ============================================================= Session States ===================================================================
if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = graph.retrieve_all_thread()

if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = {}

# Initialize history for this thread
if thread_id not in st.session_state["chat_histories"]:
    state = app.get_state(config=CONFIG)
    st.session_state["chat_histories"][thread_id] = state.values.get("message", []) or []

current_history = st.session_state["chat_histories"][thread_id]

# ============================================================ Display History ===================================================================
for message in current_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================ Main Chat Logic ===================================================================
if user_input:
    # Append user message
    current_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Update LangGraph state
    app.update_state(
        config=CONFIG,
        values={"message": current_history})
    result = None

    # Handle PDF threads
    if thread_id.startswith("PDF_"):
        filename_no_ext = thread_id.replace("PDF_", "")
        temp_pdf_path = f"./temp_{filename_no_ext}.pdf"
        input_data = {
            "documents_path": temp_pdf_path,
            "vectorstore_path": f"../Vectorstores/{filename_no_ext}/",
            "query": user_input
        }
        result = app.invoke(input_data, config=CONFIG)

    # Handle Prebuilt threads
    else:
        vectorstore_paths = {
            "Dermatology": "./vectorstores/dermatology_faiss",
            "Psychiatrist": "./vectorstores/psychiatrist_faiss",
            "Legal": "./vectorstores/legal_faiss"
        }
        input_data = {
            "vectorstore_path": vectorstore_paths[thread_id],
            "query": user_input
        }
        result = app.invoke(input_data, config=CONFIG)

    # Display AI response
    if result:
        ai_response = result["answer"]
        current_history.append({"role": "ai", "content": ai_response})
        with st.chat_message("ai"):
            st.markdown(ai_response)

        # Update LangGraph state after AI response
        app.update_state(
            config=CONFIG,
            values={"message": current_history})

    # Save updated history back
    st.session_state["chat_histories"][thread_id] = current_history

# ========================== Save Graph as PNG ==================================================================================================
graph_png = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)