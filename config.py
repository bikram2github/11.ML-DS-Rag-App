import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
#from langchain_sambanova import SambaNovaEmbeddings


load_dotenv()

#hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

'''sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
sambanova_base_url = os.getenv("SAMBANOVA_BASE_URL")'''

DATA_DIR="data"


INDEX_DIR = "faiss_index"

'''new_document_added = False

pdf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")])
if pdf_count > 10:
    new_document_added = True'''


import streamlit as st

@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

'''@st.cache_resource
def load_sambanova_embeddings():
    return SambaNovaEmbeddings(model="E5-Mistral-7B-Instruct")'''


llm = load_llm()

hf_embeddings = load_embeddings()

#sambanova_embeddings = load_sambanova_embeddings()

