'''from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
'''

#import shutil
#from config import new_document_added

import streamlit as st
from config import llm, hf_embeddings, INDEX_DIR, DATA_DIR

from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from dotenv import load_dotenv


import os

load_dotenv()



def load_document(path):
    loader = DirectoryLoader(
    path=path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
    )

    docs = loader.load()

    return docs

def text_split(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.split_documents(docs)
    return docs


def generate_embeddings(docs,DATA_DIR,INDEX_DIR,embeddings):
            
    base_name = os.path.splitext(DATA_DIR)[0]
    base_name = base_name.replace(" ", "_").replace(".", "_").replace("/", "_")
    INDEX_PATH = os.path.join(INDEX_DIR, f"{base_name}_index")

    '''if new_document_added:
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
'''
    os.makedirs(INDEX_DIR, exist_ok=True)

    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_PATH)

    return vectorstore


def generate_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return retriever


def generate_rag_chain(retriever, llm):

    qa_prompt=(
        "you are a helpful and precise AI assistant. " \
        "based on chat history rewrite the user question to be a standalone question. " \
        "if the user question is already standalone, just return the same question. " \
    )
    
    
    
    qa_rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    
    system_prompt = (
        "You are a helpful and precise AI assistant. "
        "Answer the user's question strictly based on the provided context. "   
        "IMPORTANT: Pay attention to the conversation history. If the user refers to something mentioned earlier "
        "(like 'it', 'that', 'explain more', 'what about'), use the chat history to understand the context.\n\n"     
        "if anyone ask you 'who are you', respond with 'I am an AI language model created by Bikram Maity, designed to assist with Machine Learning interview questions and related concepts.' "
        "if anyone ask you how are you, respond with 'Hello! I am doing well. How can I assist you today?' "
        "if anyone ask you hii or hello, respond with 'Hello! How can I assist you today?' "
        "If the context does not contain enough information to answer, respond with: 'I currently do not have sufficient context to respond to this question..' "
        "Do not use any external or prior knowledge. "
        "strictly If the user makes grammar mistakes, fix them in your answer. "
        "Rewrite the answer clearly using proper English and clean formatting."
        "Use numbered bullet points where appropriate."
        "Keep your answer clear, proper explainable, and directly relevant to the context below.\n\n"
        "Context:\n{context}"
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    retrievel_chain=RunnableParallel({
        "input": qa_rewrite_prompt | llm | StrOutputParser(),
        "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
    })

    parallel_chain = RunnableParallel(
        {
            "context": RunnableLambda(lambda x: "\n\n".join(d.page_content for d in retriever.invoke(x["input"]))),
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
    )

    rag_chain = retrievel_chain | parallel_chain | answer_prompt | llm | StrOutputParser()

    return rag_chain



#build rag


@st.cache_resource
def load_rag_chain(llm, embeddings,DATA_DIR, INDEX_DIR):
    docs = load_document(DATA_DIR)
    docs = text_split(docs)
    vectorstore = generate_embeddings(
        docs, DATA_DIR, INDEX_DIR, embeddings
    )
    retriever = generate_retriever(vectorstore)

    return generate_rag_chain(retriever, llm)

rag_chain = load_rag_chain(llm,hf_embeddings,DATA_DIR, INDEX_DIR)

def get_rag_chain():
    return rag_chain