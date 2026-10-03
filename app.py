import streamlit as st
from databricks.sdk import WorkspaceClient
import requests
import json

st.set_page_config(page_title="AWS Documentation Assistant", layout="centered")

st.title("AWS Documentation Assistant")
st.caption("Ask questions about AWS security, cost optimization, and best practices.")

ENDPOINT_NAME = "aws_rag_chatbot"

@st.cache_resource
def get_client():
    return WorkspaceClient()

def query_chatbot(question):
    try:
        client = get_client()
        # Use SDK for auth, but make raw HTTP call for flexibility
        host = client.config.host.rstrip("/")
        headers = client.config.authenticate()
        headers["Content-Type"] = "application/json"
        
        url = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
        payload = {"messages": [{"role": "user", "content": question}]}
        
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        
        if r.status_code != 200:
            return f"HTTP {r.status_code}: {r.text[:400]}"
        
        data = r.json()
        
        # Handle all possible response formats
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                if "choices" in first:
                    return first["choices"][0]["message"]["content"]
                return f"List item format: {str(first)[:300]}"
        
        if isinstance(data, dict):
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            if "predictions" in data:
                p = data["predictions"]
                if isinstance(p, list) and p:
                    if isinstance(p[0], dict) and "choices" in p[0]:
                        return p[0]["choices"][0]["message"]["content"]
        
        return f"Unknown format: {str(data)[:400]}"
    
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Options")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Endpoint: aws_rag_chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your AWS question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching AWS documentation..."):
            answer = query_chatbot(user_input)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})