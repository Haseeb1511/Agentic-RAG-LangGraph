import streamlit as st
import requests
from pathlib import Path
from typing import Dict
import os,sys

# Add project root (one level up from frontend) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio.audio_record import speech_to_text,record_audio


# ============================================= Configuration ==========================================

# FastAPI backend URL
# API_BASE_URL = "http://localhost:8000"
API_BASE_URL = os.environ.get("API_URL", "http://localhost:8000")  # for server (aws) or docker compose this work

#================================================= Helper Functions =============================================
def make_api_request(endpoint:str,method:str="GET",data:dict=None,files:Dict=None):
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}" #remove leading whitespaces
    if method == "GET":
        response = requests.get(url)
    elif method == "POST":
        if files:
            response = requests.post(url,files=files)
        else:
            response = requests.post(url,json=data)
    else:
        raise ValueError(f"Unsupported method:{method} ")
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API error: {response.status_code}")
        return {}


def load_vactorstores():
    """Load Avaliable Vectorstores from the API"""
    response = make_api_request("/vectorstores")
    if response:
        return ["Landing Page"] + response.get("all_choices",[])
    return ["Landing Page","Dermatology🩺", "Psychiatrist🧠", "Legal🏛️"] #fall back


def upload_pdf_to_api(uploaded_file):
    """Upload PDF to FastAPI backend"""
    #we need to send it in exact same format required by Fast API UploadFile
    files = {"file":(uploaded_file.name, uploaded_file.getvalue(),"application/pdf")}
    response = make_api_request("/upload_pdf",method="POST",files=files)
    return response


def send_query_to_api(query:str,thread_id:str):
    """SEnd query to ABckend"""
    data = {"query":query,
            "thread_id":thread_id}
    response = make_api_request(endpoint="/query",method="POST",data=data)
    return response


def load_chat_history(thread_id:str):
    """Load chat histroy from backend"""
    response = make_api_request(f"chat_history/{thread_id}")
    if response:
        return response.get("messages",[])
    return []


#============================================= Streamlit Configuration ===========================================

st.set_page_config(
    page_title="Agentic RAG Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


#================================================ Session State ================================================
if "uploaded_pdfs" not in st.session_state:
    st.session_state["uploaded_pdfs"] = []

if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = "Landing Page" #default we will show landing page

if "api_connected" not in st.session_state:
    health_Check = make_api_request("/")
    st.session_state["api_connected"] = bool(health_Check)


# ========================================= Sidebar ======================================================

st.sidebar.title("🤖 Agentic RAG Q&A")

#API status
if st.session_state["api_connected"]:
    st.sidebar.success("API Connected")
else:
    st.sidebar.error("API DIsconnected")

# ============================================= Upload PDF ======================================================
st.sidebar.header("📄 Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload pDF to chat with them",type=["PDF"])
if uploaded_pdf and st.session_state["api_connected"]:
    filename_no_ext = Path(uploaded_pdf.name).stem
    pdf_choice = f"PDF_{filename_no_ext}"

    if pdf_choice not in st.session_state["uploaded_pdfs"]:
        with st.spinner("Uploading PDF...."):
            uploaded_response = upload_pdf_to_api(uploaded_file=uploaded_pdf)

        if uploaded_response:
            st.session_state["uploaded_pdfs"].append(pdf_choice)
            st.sidebar.success(f"{uploaded_pdf.name} uploaded!")
        else:
            st.sidebar.error("Failed to Upload PDF")

# ========================================= load Avaliable Choices after upload PDF ======================
#Load avaliable chice
if st.session_state["api_connected"]:
    all_choices = load_vactorstores()
else:
    all_choices = ["Landing Page", "Dermatology🩺", "Psychiatrist🧠", "Legal🏛️"] #fallback

#choice selection
st.sidebar.header("Chose Knowledge Base")
choice=  st.sidebar.radio("Select from",all_choices)

#===================================== Make sure that current thread is same as Choice ===================

# Update current thread when choice changes
if choice != st.session_state["current_thread"]:
    st.session_state["current_thread"] = choice


#========================================================== Title ============================================

st.title("🤖 Agentic RAG Q&A")

#============================================== landing Page ==========================================
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

# =============================================== Load Chat History =============================================
# Only show chat if API is connected and not on landing page
if not st.session_state["api_connected"]:
    st.error("🚫 Cannot start chat without API connection. Please start the FastAPI backend.")
    st.stop()

# Display current knowledge base
st.subheader(f"💬 Chatting with: {choice}")

thread_id = choice
messages = load_chat_history(thread_id=thread_id)

for message in messages:
    role  = message["role"] if isinstance(message,dict) else message.role
    content = message["content"] if isinstance(message,dict) else message.content

    with st.chat_message(role):
        st.markdown(content)

 # ====================================================== User input =============================================

text_input = st.chat_input("Ask anything ....")

if st.button("🎤 Speak"):
    with st.spinner("listining..."):
        audio_path = record_audio()
        audio_input = speech_to_text(audio_path=audio_path)
else:
    audio_input=None

user_input = text_input or audio_input

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Send query to API and get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking...."):
            response = send_query_to_api(query=user_input,thread_id=thread_id)

        if response and "answer" in response:
            st.markdown(response["answer"])
        else:
            st.error("Sorry i could not process your query")

# ========================================== Sidebar Additional Options ===================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Options")

# Refresh button
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# # Clear chat button
# if st.sidebar.button("🗑️ Clear Chat History"):
#     st.warning("Chat history clearing not implemented yet")

