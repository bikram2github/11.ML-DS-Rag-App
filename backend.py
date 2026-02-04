import os
import streamlit as st
from rag_chatbot import get_rag_chain 

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,AIMessage


from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

#import sqlite3
#from langgraph.checkpoint.sqlite import SqliteSaver
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver


dbname=os.getenv("PG_DB")
db_user=os.getenv("PG_USER")
db_password=os.getenv("PG_PASSWORD")
db_host=os.getenv("PG_HOST")
db_port=os.getenv("PG_PORT")




#add rag_chain
try:
    rag_chain=get_rag_chain()
except Exception as e:
    raise Exception("RAG chain could not be loaded")




#Langgraph PArt

#State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def ask(state: ChatState):
    question = state["messages"][-1].content
    chat_history = state["messages"]
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    return {
        "messages": [AIMessage(content=response)]
    }


"""def chat_node(state:ChatState):
    message=state["messages"]
    response=llm.invoke(message)

    return {"messages": [response]}"""

#conn=sqlite3.connect(database="chatbot1.db",check_same_thread=False)


@st.cache_resource
def get_pg_connection():
    return psycopg.connect(
        dbname=dbname,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        sslmode="require",
        autocommit=True
    )

conn = get_pg_connection()


checkpoint_saver=PostgresSaver(conn)
checkpoint_saver.setup()

graph=StateGraph(ChatState)

graph.add_node("rag",ask)

graph.add_edge(START,"rag")
graph.add_edge("rag",END)

chatbot=graph.compile(checkpointer=checkpoint_saver)

def get_connection():
    return conn

def get_chatbot():
    return chatbot


    
def retrieve_all_history():
    all_threads=set()
    for checkpoint in checkpoint_saver.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)


'''response=chatbot.invoke({"messages":[HumanMessage(content="Hello")]},config={"configurable":{"thread_id":"test_once"}})

print(response)'''