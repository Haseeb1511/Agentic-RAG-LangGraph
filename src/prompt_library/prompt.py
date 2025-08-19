from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(template = """
You are a helpful assistant.


Here are the relevant documents:
{context}

Now answer the user's question:
{question}
""",
input_variables=["context", "question"]
)

