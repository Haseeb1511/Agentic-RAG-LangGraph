from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


# setting up ENV variable
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()


#load .env from root folder
# env_path = Path(__file__).resolve().parents[1] / ".env"
# load_dotenv(dotenv_path=env_path)

# groq_api = os.getenv("GROQ_API_KEY")
# openai_api = os.getenv("OPENAI_API_KEY")



model = ChatOpenAI(model="gpt-4.1-nano")
model = ChatGroq(model="Llama-3.3-70B-Versatile")

print("Success")

