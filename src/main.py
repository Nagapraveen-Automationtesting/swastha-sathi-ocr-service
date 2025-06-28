from fastapi import FastAPI
from src.routes.upload import router as upload_router
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS




app = FastAPI()
app.include_router(upload_router)

print("Basic LangChain setup is working.")
