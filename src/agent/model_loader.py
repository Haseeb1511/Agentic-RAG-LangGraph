from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_cohere import CohereRerank
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq
import os

# setting up ENV variable
# from dotenv import load_dotenv
# from pathlib import Path
# import os
# env_path = Path(__file__).resolve().parents[1] / ".env"
# load_dotenv(dotenv_path=env_path)

# groq_api = os.getenv("GROQ_API_KEY")
# openai_api = os.getenv("OPENAI_API_KEY")


from dotenv import load_dotenv
load_dotenv()


import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# Brain
model = ChatGroq(model="Llama-3.3-70B-Versatile")

reranker_llm = CohereRerank(model="rerank-english-v3.0") 

summary_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# def Embedder():
#     return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

EMBEDDER = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

audio_converter_model = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("Success")

