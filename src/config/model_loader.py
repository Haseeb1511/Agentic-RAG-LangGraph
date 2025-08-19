from src.config.config_loader import load_config


from pydantic import BaseModel,Field
from typing import TypedDict,Literal,Any,Optional
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq


from dotenv import load_dotenv
from pathlib import Path
import os,sys
#load .env from root folder
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

groq_api = os.getenv("GROQ_API_KEY")
openai_api = os.getenv("OPENAI_API_KEY")


class ConfigLoader:
    def __init__(self):
        self.config = load_config()

    def __getitem__(self,key):
        return self.config[key]    #we give key and get corresponsidng value from config.yaml like we give llm name and it get specific model
    


class ModelLoader(BaseModel):
    model_provider:Literal["groq","openai"]="groq"
    config:Optional[ConfigLoader]=Field(default=None,exclude=True)

    def model_post_init(self,__context:Any):
        self.config = ConfigLoader()

    class Config:
        arbitrary_types_allowed = True  #Allows you to store non-primitive Python objects in your Pydantic model without validation errors.
    
    def load_llm(self):
        if self.model_provider=="openai":
            model_name = self.config["llm"]["openai"]["model_name"]
            llm = ChatOpenAI(model=model_name,api_key=openai_api)
        elif self.model_provider=="groq":
            model_name = self.config["llm"]["groq"]["model_name"]
            llm= ChatGroq(model=model_name,api_key=groq_api)
        return llm

    