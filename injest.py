import os
from dotenv import load_env
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_env()
def run_ingestion():
    api_key=os.env("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    #load PDF
    loader=PyPDFLoader("data/notes.pdf")
    documents=loader.load()
    
    # SPLIT documents
    splitter = RecursiveCharacterTextSplitter(chunk=1000 , chunk_overlap=200)
    chunks=splitter.split_documents(documents)
    
    # Use gemini-embedding-2 for consistent vector space
    embeddings=GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key
        )
    
    # Create Chroma DB (Ensure ./chroma_db is deleted before running)
    vectortore= Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db")
    
    print(f"Success!! Processed {len(chunks)} chunks.")
    
    if __name__=="__main__":
        run_ingestion()