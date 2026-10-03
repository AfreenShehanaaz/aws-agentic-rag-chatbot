# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Chatbot Notebook
# MAGIC
# MAGIC This notebook implements the retrieval-augmented generation pipeline.
# MAGIC It queries the vector search index built in notebook 01 and generates
# MAGIC answers using a hosted chat model.

# COMMAND ----------

# MAGIC %pip install --quiet langchain langchain-community databricks-vectorsearch databricks-langchain mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

VECTOR_ENDPOINT = "rag_endpoint"
VECTOR_INDEX = "workspace.default.pdf_chunks_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
TOP_K = 3

# COMMAND ----------

from databricks_langchain import DatabricksVectorSearch

vector_store = DatabricksVectorSearch(
    index_name="workspace.default.pdf_chunks_index",
    columns=["source_file", "page"]
)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

test_docs = retriever.invoke("What are AWS security best practices?")
for i, d in enumerate(test_docs, 1):
    print(f"[{i}] {d.metadata.get('source_file')} p.{d.metadata.get('page')}")
    print(d.page_content[:200])
    print()

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    "You are an AWS documentation assistant. Answer the question using only "
    "the context below. If the context does not contain the answer, say you "
    "do not have that information. Cite the source file and page at the end.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=800)

def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source_file')} p.{d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain built.")

# COMMAND ----------

questions = [
    "What are AWS security best practices?",
    "How can I optimize AWS costs?",
    "What is the AWS Well-Architected Framework?"
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {rag_chain.invoke(q)}")
    print("-" * 80)

# COMMAND ----------

import mlflow
import time

mlflow.set_experiment("/Users/shehanaazafreen57@gmail.com/aws_rag_experiments")

def ask_and_log(question):
    start = time.time()
    with mlflow.start_run(run_name="query", nested=False):
        retrieved = retriever.invoke(question)
        chunks_info = [
            {"source": d.metadata.get("source_file"), 
             "page": d.metadata.get("page"),
             "preview": d.page_content[:150]}
            for d in retrieved
        ]
        answer = rag_chain.invoke(question)
        latency = time.time() - start

        mlflow.log_param("question", question)
        mlflow.log_param("num_chunks_retrieved", len(retrieved))
        mlflow.log_param("llm_endpoint", LLM_ENDPOINT)
        mlflow.log_param("top_k", TOP_K)
        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_dict({"chunks": chunks_info}, "retrieved_chunks.json")
        mlflow.log_text(answer, "response.txt")

        return answer, latency

# Test with tracking
q = "What is AWS Well-Architected Framework?"
ans, lat = ask_and_log(q)
print(f"Latency: {lat:.2f}s")
print(f"Answer: {ans[:300]}...")

# COMMAND ----------

import os

os.makedirs("/tmp/rag_agent", exist_ok=True)

# Write chain code directly to file using triple quotes properly
with open("/tmp/rag_agent/chain.py", "w") as f:
    f.write("""from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import mlflow

VECTOR_ENDPOINT = "rag_endpoint"
VECTOR_INDEX = "workspace.default.pdf_chunks_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
TOP_K = 3

vector_store = DatabricksVectorSearch(
    endpoint=VECTOR_ENDPOINT,
    index_name=VECTOR_INDEX,
    columns=["source_file", "page"]
)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

prompt = ChatPromptTemplate.from_template(
    "You are an AWS documentation assistant. Answer the question using only "
    "the context below. If the context does not contain the answer, say you "
    "do not have that information. Cite the source file and page at the end.\\n\\n"
    "Context:\\n{context}\\n\\n"
    "Question: {question}\\n\\n"
    "Answer:"
)

llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=800)

def format_docs(docs):
    parts = []
    for d in docs:
        src = d.metadata.get("source_file")
        pg = d.metadata.get("page")
        parts.append(f"[{src} p.{pg}]\\n{d.page_content}")
    return "\\n\\n".join(parts)

def extract_question(chat_request):
    messages = chat_request.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""

def format_response(answer_text):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": answer_text},
            "finish_reason": "stop"
        }]
    }

chain = (
    RunnableLambda(extract_question)
    | {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(format_response)
)

mlflow.models.set_model(chain)
""")

print("Chain file saved successfully.")

# Verify by printing the file contents
with open("/tmp/rag_agent/chain.py", "r") as f:
    print("--- File contents ---")
    print(f.read())

# COMMAND ----------

import mlflow
from mlflow.models.rag_signatures import ChatCompletionRequest, ChatCompletionResponse
from mlflow.models.signature import ModelSignature

mlflow.set_registry_uri("databricks-uc")

MODEL_NAME = "workspace.default.aws_rag_chatbot"

signature = ModelSignature(
    inputs=ChatCompletionRequest(),
    outputs=ChatCompletionResponse()
)

input_example = {
    "messages": [
        {"role": "user", "content": "What is AWS IAM?"}
    ]
}

with mlflow.start_run(run_name="aws_rag_deployment_v2") as run:
    logged_model = mlflow.langchain.log_model(
        lc_model="/tmp/rag_agent/chain.py",
        artifact_path="chain",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "databricks-langchain",
            "langchain",
            "langchain-community",
            "mlflow"
        ]
    )

print(f"Model registered: {MODEL_NAME}")
print(f"Version: {logged_model.registered_model_version}")

# COMMAND ----------

from mlflow.deployments import get_deploy_client

client = get_deploy_client("databricks")

endpoint_name = "aws_rag_chatbot"

client.create_endpoint(
    name=endpoint_name,
    config={
        "served_entities": [{
            "name": "aws_rag_chatbot",
            "entity_name": "workspace.default.aws_rag_chatbot",
            "entity_version": "2",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
    }
)

print(f"Endpoint created: {endpoint_name}")
print(f"URL: https://dbc-d72b57dc-7357.cloud.databricks.com/serving-endpoints/{endpoint_name}")

# COMMAND ----------

import requests
import json

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

ENDPOINT_URL = f"{host}/serving-endpoints/aws_rag_chatbot/invocations"
HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def chat(question):
    payload = {"messages": [{"role": "user", "content": question}]}
    r = requests.post(ENDPOINT_URL, headers=HEADERS, data=json.dumps(payload))
    
    if r.status_code != 200:
        return f"Error {r.status_code}: {r.text}"
    
    result = r.json()
    
    # Debug: print the raw response structure first time
    if not hasattr(chat, "_debugged"):
        print("Raw response structure:")
        print(json.dumps(result, indent=2)[:500])
        print("---")
        chat._debugged = True
    
    # Try multiple response formats
    try:
        # Format 1: OpenAI-compatible chat completion
        return result["choices"][0]["message"]["content"]
    except (KeyError, TypeError):
        pass
    
    try:
        # Format 2: Predictions wrapper (MLflow standard)
        preds = result["predictions"]
        if isinstance(preds, list):
            first = preds[0]
            if isinstance(first, dict):
                if "choices" in first:
                    return first["choices"][0]["message"]["content"]
                return str(first)
            return str(first)
        return str(preds)
    except (KeyError, TypeError):
        pass
    
    try:
        # Format 3: Direct list of choices
        if isinstance(result, list):
            return result[0]["message"]["content"]
    except (KeyError, TypeError):
        pass
    
    # Fallback: return raw JSON
    return json.dumps(result, indent=2)

print("Chatbot ready. Testing with a sample question...\n")
print(chat("What are AWS security best practices?"))

# COMMAND ----------

#  Enable MLflow autologging for automatic tracing
import mlflow

mlflow.langchain.autolog()
print("MLflow autologging enabled for LangChain.")

# Test: Run 3 queries - each will be auto-traced
test_questions = [
    "What are AWS security best practices?",
    "How can I reduce AWS costs?",
    "What is the Well-Architected Framework?"
]

for q in test_questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer[:250]}...")

print("\n" + "=" * 60)
print("All queries above are now auto-logged as MLflow traces!")
print("Check Experiments page to see them.")

# COMMAND ----------

# MAGIC %pip install --quiet langchain-classic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks_langchain import DatabricksVectorSearch, ChatDatabricks
import mlflow

vector_store = DatabricksVectorSearch(
    endpoint="rag_endpoint",
    index_name="workspace.default.pdf_chunks_index",
    columns=["source_file", "page"]
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-3-70b-instruct", temperature=0.1, max_tokens=800)

mlflow.langchain.autolog()
mlflow.set_experiment("/Users/shehanaazafreen57@gmail.com/aws_rag_experiments")
print("Ready.")

# COMMAND ----------

from langchain_core.tools import tool

@tool
def search_aws_docs(query: str) -> str:
    """Search AWS documentation for information about a specific topic."""
    docs = retriever.invoke(query)
    result = ""
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source_file", "unknown")
        pg = d.metadata.get("page", "?")
        result += f"[Source: {src}, page {pg}]\n{d.page_content}\n\n"
    return result

@tool
def get_document_source(source_file: str) -> str:
    """Get information about a specific AWS documentation source file."""
    sources = {
        "aws-overview.pdf": "Overview of AWS services (162 pages)",
        "wellarchitected-security-pillar.pdf": "Security best practices (240 pages)",
        "wellarchitected-cost-optimization-pillar.pdf": "Cost optimization (155 pages)"
    }
    return sources.get(source_file, f"Available: {', '.join(sources.keys())}")

@tool
def expand_aws_acronym(acronym: str) -> str:
    """Expand common AWS service acronyms to their full names."""
    acronyms = {
        "IAM": "Identity and Access Management",
        "EC2": "Elastic Compute Cloud",
        "S3": "Simple Storage Service",
        "VPC": "Virtual Private Cloud",
        "RDS": "Relational Database Service",
        "MFA": "Multi-Factor Authentication",
        "KMS": "Key Management Service"
    }
    return acronyms.get(acronym.upper().strip(), f"'{acronym}' not in database")

@tool
def list_available_topics() -> str:
    """List AWS topics the chatbot knows about."""
    return "Available: Security (IAM, encryption), Cost Optimization, AWS Overview"

tools = [search_aws_docs, get_document_source, expand_aws_acronym, list_available_topics]
print(f"{len(tools)} tools defined.")

# COMMAND ----------

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AWS documentation assistant with access to tools.

- Use search_aws_docs for any AWS technical question
- Use expand_aws_acronym for acronyms like IAM, EC2, S3
- Use list_available_topics for meta questions
- Use get_document_source for source questions
- Always cite the source file and page number
- Remember previous conversation context"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

print("Agent ready with 4 tools + memory.")

# COMMAND ----------

session_config = {"configurable": {"session_id": "test-1"}}

for q in ["What is IAM?", "How do I use it to secure my account?", "What tools help with this?"]:
    print("=" * 70)
    print(f"Q: {q}")
    print("=" * 70)
    response = agent_with_memory.invoke({"input": q}, config=session_config)
    print(f"A: {response['output']}\n")