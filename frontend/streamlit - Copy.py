import sys,os
# Add project root to sys.path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.agent.agentic_workflow import GraphBuilder
from pathlib import Path

from langchain_core.messages import HumanMessage,AIMessage

# ================================================================ Build Graph Once ==============================================================
if "graph_builder" not in st.session_state:
    st.session_state.graph_builder = GraphBuilder()
if "graph_app" not in st.session_state:
    st.session_state.graph_app = st.session_state.graph_builder.build_graph()

graph = st.session_state.graph_builder
app = st.session_state.graph_app

# =========================================================== Page Title =========================================================================
st.title("Agentic RAG Q&A")

# ======================================================= User Choice + PDFs Handling ============================================================

# choice list
base_choices = ["Landing Page","Dermatology🩺", "Psychiatrist🧠", "Legal🏛️"]

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


# ================================================================Landing Page==================================================================
# Show landing page if selected
if choice == "Landing Page":
    st.markdown("<h1 style='text-align: center; color:red;'>Agentic RAG Q&A</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #6A5ACD;'>Chat intelligently with predefined knowledge bases or your own PDFs</h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### How to use the app:")
    st.markdown("""
    1. Select a vector store from the sidebar or upload a PDF.
    2. Start asking questions once a store is selected.
    """)

    st.markdown("### Explore Predefined Knowledge Bases:")

    # Use columns to make it look like cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div style='text-align:center; font-size:30px;'>🩺</div>", unsafe_allow_html=True)
        st.markdown("**Dermatology**")
        st.markdown("Learn about skin conditions, treatments, and health advice.")

    with col2:
        st.markdown("<div style='text-align:center; font-size:30px;'>🧠</div>", unsafe_allow_html=True)
        st.markdown("**Psychiatrist**")
        st.markdown("Explore mental health, psychology, and therapy-related info.")

    with col3:
        st.markdown("<div style='text-align:center; font-size:30px;'>🏛️</div>", unsafe_allow_html=True)
        st.markdown("**Legal**")
        st.markdown("Access legal knowledge, laws, and regulations.")

    st.markdown("---")

    st.markdown("### Upload Your Own PDF")
    st.markdown("You can upload any PDF document and chat with it intelligently!")

    # Stop execution so chat input doesn't show
    st.stop()


#======================================================= User Text Input =========================================================================
user_input = st.chat_input("Ask Anything....")

# use choice as thread_id
thread_id = choice
CONFIG = {"configurable": {"thread_id": thread_id}}

# ============================================================= Session States ===================================================================
# It retrieve all chat messages from the "./chat_hist/chat.db"
# graph.retrieve_all_thread() fetches all saved chat threads from the SQLite database where LangGraph stores them.
if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = graph.retrieve_all_thread()

# ============================================================ Display History ===================================================================

#On restarting the Streamlit server, LangGraph reloads the messages from this SQLite database
def load_conversation(thread_id):
    """It reads from the LangGraph app in-memory state for a specific thread.
     Loads the chat history for a specific conversation thread.
    Note:
        This reads from the app's current in-memory state, 
        not directly from the database.
    """
    state = app.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

messages = load_conversation(thread_id)
for message in messages:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "system"

    with st.chat_message(role):
        st.markdown(message.content)


# ============================================================ Main Chat Logic ===================================================================

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

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

    # Handle Prebuilt VectorStore
    else:
        vectorstore_paths = {
            "Dermatology🩺": "./vectorstores/dermatology_faiss",
            "Psychiatrist🧠": "./vectorstores/psychiatrist_faiss",
            "Legal🏛️": "./vectorstores/legal_faiss"
        }
        input_data = {
            "vectorstore_path": vectorstore_paths[thread_id],
            "query": user_input
        }
        result = app.invoke(input_data, config=CONFIG)

    # Display AI response
    if result:
        ai_response = result["answer"]

        with st.chat_message("assistant"):
            st.markdown(ai_response)

# ========================== Save Graph as PNG ==================================================================================================
graph_png = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)
