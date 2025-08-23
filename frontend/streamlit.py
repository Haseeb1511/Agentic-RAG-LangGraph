import streamlit as st
import requests
import json
from pathlib import Path
from typing import List, Dict, Any


# ================================================================ Configuration ==============================================================

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"


# ================================================================ Helper Functions ==============================================================

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None):
    """Make API request to FastAPI backend"""
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
        
    if method == "GET":# for predefined vector store
        response = requests.get(url)
    elif method == "POST": #for pdf
        if files:
            response = requests.post(url, files=files)  # for pdf post
        else:
            response = requests.post(url, json=data)
    else:
        raise ValueError(f"Unsupported method: {method}")
        
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return {}
    


def load_vectorstores():
    """Load available vectorstores from API"""
    response = make_api_request("/vectorstores")
    if response:
        return  ["Landing Page"] + response.get("all_choices", [])
    return ["Landing Page","Dermatology🩺", "Psychiatrist🧠", "Legal🏛️"]  # Fallback


def upload_pdf_to_api(uploaded_file):
    """Upload PDF to FastAPI backend"""
    #we need to send it in exact same format required by Fast API UploadFile
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")} 
    response = make_api_request("/upload-pdf", method="POST", files=files)
    return response


def send_query_to_api(query: str, thread_id: str):
    """Send query to FastAPI backend"""
    data = {
        "query": query,
        "thread_id": thread_id
    }
    response = make_api_request("/query", method="POST", data=data)
    return response

def load_chat_history(thread_id: str):
    """Load chat history from API"""
    response = make_api_request(f"/chat-history/{thread_id}")
    if response:
        return response.get("messages", [])
    return []


# ============================================================= Streamlit Configuration ==========================================================

st.set_page_config(
    page_title="Agentic RAG Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================ Session State ==============================================================

# Initialize session states
if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = []
if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = "Landing Page"
if "api_connected" not in st.session_state:
    # Test API connection
    health_check = make_api_request("/")
    st.session_state["api_connected"] = bool(health_check)

# ================================================================ Sidebar ==============================================================

st.sidebar.title("🤖 Agentic RAG Q&A")

# API Status
if st.session_state["api_connected"]:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.write("Please start FastAPI server: `uvicorn app:app --reload`")

# PDF Upload
st.sidebar.header("📄 Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF here to Chat with it", type=['pdf'])

if uploaded_pdf and st.session_state["api_connected"]:
    filename_no_ext = Path(uploaded_pdf.name).stem
    pdf_choice = f"PDF_{filename_no_ext}"
    
    if pdf_choice not in st.session_state["uploaded_pdfs"]:
        with st.spinner("Uploading PDF..."):
            upload_response = upload_pdf_to_api(uploaded_pdf)
            
        if upload_response:
            st.session_state["uploaded_pdfs"].append(pdf_choice)
            st.sidebar.success(f"✅ {uploaded_pdf.name} uploaded!")
        else:
            st.sidebar.error("❌ Failed to upload PDF")


# Load available choices
if st.session_state["api_connected"]:
    all_choices = load_vectorstores()
else:
    all_choices = ["Landing Page", "Dermatology🩺", "Psychiatrist🧠", "Legal🏛️"]

# Choice selection
st.sidebar.header("📂Choose Knowledge Base")
choice = st.sidebar.radio("Select from:", all_choices)

# Update current thread when choice changes
if choice != st.session_state["current_thread"]:
    st.session_state["current_thread"] = choice

# ================================================================ Main Page ==============================================================

# Page Title
st.title("🤖 Agentic RAG Q&A")

# Landing Page
if choice == "Landing Page":
    st.markdown("<h1 style='text-align: center; color:red;'>Agentic RAG Q&A</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #6A5ACD;'>Chat intelligently with predefined knowledge bases or your own PDFs</h3>", unsafe_allow_html=True)
    st.markdown("---")

    # Architecture Info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        - **FastAPI Backend**: Handles AI processing and data management
        - **Streamlit Frontend**: Provides user interface
        - **Agentic RAG**: Intelligent document retrieval and generation
        """)
    
    with col2:
        st.markdown("### 🚀 How to use:")
        st.markdown("""
        1. Ensure FastAPI server is running
        2. Select a knowledge base or upload PDF
        3. Start asking questions
        """)

    st.markdown("---")
    st.markdown("### 📚 Explore Predefined Knowledge Bases:")

    # Knowledge base cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='text-align:center; padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 10px;'>
            <div style='font-size:50px;'>🩺</div>
            <h3>Dermatology</h3>
            <p>Learn about skin conditions, treatments, and health advice.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 10px;'>
            <div style='font-size:50px;'>🧠</div>
            <h3>Psychiatrist</h3>
            <p>Explore mental health, psychology, and therapy-related info.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align:center; padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 10px;'>
            <div style='font-size:50px;'>🏛️</div>
            <h3>Legal</h3>
            <p>Access legal knowledge, laws, and regulations.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📎 Upload Your Own PDF")
    st.markdown("You can upload any PDF document and chat with it intelligently using the sidebar!")
    
    # API Status Details
    if st.session_state["api_connected"]:
        st.success("🟢 FastAPI Backend is connected and ready!")
    else:
        st.error("🔴 FastAPI Backend is not running. Please start it with: `uvicorn main:app --reload`")

    st.stop()

# ================================================================ Chat Interface ==============================================================

# Only show chat if API is connected and not on landing page
if not st.session_state["api_connected"]:
    st.error("🚫 Cannot start chat without API connection. Please start the FastAPI backend.")
    st.stop()

# Display current knowledge base
st.subheader(f"💬 Chatting with: {choice}")

# Load and display chat history
thread_id = choice
messages = load_chat_history(thread_id)

for message in messages:
    role = message["role"] if isinstance(message, dict) else message.role
    content = message["content"] if isinstance(message, dict) else message.content
    
    with st.chat_message(role):
        st.markdown(content)

# Chat input
user_input = st.chat_input("Ask anything...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Send query to API and get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_query_to_api(user_input, thread_id)
        
        if response and "answer" in response:
            st.markdown(response["answer"])
        else:
            st.error("Sorry, I couldn't process your question. Please try again.")

# ================================================================ Sidebar Additional Options ==============================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Options")

# Refresh button
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.warning("Chat history clearing not implemented yet")

# # Show uploaded PDFs
# if st.session_state["uploaded_pdfs"]:
#     st.sidebar.header("📎 Uploaded PDFs")
#     for pdf in st.session_state["uploaded_pdfs"]:
#         pdf = pdf.replace("PDF_", "")
#         st.sidebar.text(f"• {pdf}")

# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("""
# <div style='text-align: center; color: #666; font-size: 12px;'>
#     Made with ❤️ using FastAPI + Streamlit by Haseeb
# </div>
# """, unsafe_allow_html=True)