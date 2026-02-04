
from backend import get_chatbot,retrieve_all_history,get_connection
from langchain_core.messages import HumanMessage,AIMessage
import streamlit as st
import uuid


try:
    
    chatbot=get_chatbot()
    conn=get_connection()

except Exception as e:
    raise Exception("Could not initialize chatbot or database connection. Please check backend.py for errors.")


def full_reset_app():
    st.session_state.clear()

    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE checkpoint_blobs, checkpoint_writes, checkpoints CASCADE;"
        )

    st.rerun()





def generate_thread_id():
    thread_id= uuid.uuid4()
    return str(thread_id)



def new_chat():
    thread_id=generate_thread_id()
    st.session_state.thread_id=thread_id
    st.session_state.message_history=[]
    add_thread(st.session_state.thread_id)


def add_thread(thread_id):
    if thread_id not in st.session_state.all_threads:
        st.session_state.all_threads.append(thread_id)


def load_conversation(thread_id):
    state=chatbot.get_state(config={"configurable":{"thread_id":thread_id}})
    return state.values.get("messages",[])



def rename_threads(thread_id):
    messages=load_conversation(thread_id)
    title=""   
    if not messages:
        title="Start Chatting"
    else:
        title=messages[0].content

    title=title[:40]+"..." if len(title) > 40 else title    
    return title


if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "all_threads" not in st.session_state:
    st.session_state.all_threads = retrieve_all_history()

add_thread(st.session_state.thread_id)


st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    new_chat()

if st.sidebar.button("Reset All History"):
    full_reset_app()


st.sidebar.header("Chat history")

for thread_id in st.session_state.all_threads[::-1]:
    if st.sidebar.button(label=str(rename_threads(thread_id)).title(),key=f"thread_btn_{thread_id}"):
        st.session_state.thread_id=thread_id
        messages=load_conversation(thread_id)
        temp_message=[]

        for msg in messages:
            if isinstance(msg,HumanMessage):
                role="user"
            
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                # Fallback for unknown message types
                role = "assistant"            

            temp_message.append({"role":role,"message":msg.content})
        st.session_state.message_history=temp_message


CONFIG={"configurable":{"thread_id":st.session_state.thread_id}}


for message in st.session_state.message_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["message"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["message"])



user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.message_history.append(
        {"role": "user", "message": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG
        )

        if isinstance(result, dict) and "messages" in result:
            ai_message = result["messages"][-1].content
        else:
            ai_message = str(result)

        st.markdown(ai_message)

    st.session_state.message_history.append(
        {"role": "assistant", "message": ai_message}
    )

