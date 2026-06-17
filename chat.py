import os 
import streamlit as st

# We do not need dotenv anymore! Streamlit handles secrets automatically.

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Pull the key safely from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]

# page config
st.set_page_config(
    page_title="Study Buddy",
    page_icon="📚",
    layout="wide"
)

st.title("Study Buddy")
st.write("Ask questions from your notes")

# load models
@st.cache_resource
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    llm = ChatGoogleGenerativeAI( 
        model="gemini-3.5-flash",
        google_api_key=api_key
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful study assistant.
    Answer ONLY from the provided context.

    Context:
    {context}

    Question:
    {input}""")
    
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain
    )
    
    return retrieval_chain

retrieval_chain = load_rag_chain()

# CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# user input
user_query = st.chat_input("Ask anything from your notes")

if user_query:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_query})
        
        # Fixed typo: changed "amswer" to "answer"
        answer = response["answer"]
        st.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
