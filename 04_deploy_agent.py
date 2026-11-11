# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Agent deployment
# MAGIC
# MAGIC Registers the agent in Unity Catalog and serves it on `aws_rag_chatbot`.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain databricks-vectorsearch databricks-sdk langchain langchain-core langchain-community langchain-classic mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Config
from databricks.sdk import WorkspaceClient

VECTOR_ENDPOINT = "rag_endpoint"
VECTOR_INDEX = "workspace.default.pdf_chunks_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_ENDPOINT = "databricks-qwen3-embedding-0-6b"
MODEL_NAME = "workspace.default.aws_rag_chatbot"
SERVING_ENDPOINT = "aws_rag_chatbot"

w = WorkspaceClient()

USER = w.current_user.me().user_name
EXPERIMENT = f"/Users/{USER}/aws_rag_experiments"

print("Vector Search endpoints:", [e.name for e in w.vector_search_endpoints.list_endpoints()])
print("Serving endpoints:", [e.name for e in w.serving_endpoints.list()])

# COMMAND ----------

# Agent code for serving
import os

os.makedirs("/tmp/rag_agent", exist_ok=True)

with open("/tmp/rag_agent/agent.py", "w") as f:
    f.write(r'''
from databricks_langchain import ChatDatabricks, DatabricksVectorSearch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
import mlflow

VECTOR_ENDPOINT = "rag_endpoint"
VECTOR_INDEX = "workspace.default.pdf_chunks_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
TOP_K = 3

vector_store = DatabricksVectorSearch(
    endpoint=VECTOR_ENDPOINT,
    index_name=VECTOR_INDEX,
    columns=["source_file", "page"],
)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K, "query_type": "HYBRID"})
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=800)


@tool
def search_aws_docs(query: str) -> str:
    """Search AWS documentation for information about a specific topic."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant AWS documentation was found."
    return "\n\n".join(
        f"[{d.metadata.get('source_file', 'unknown')} p.{int(d.metadata.get('page', 0))}]\n{d.page_content}"
        for d in docs
    )


@tool
def get_document_source(source_file: str) -> str:
    """Get information about a specific AWS documentation source file."""
    sources = {
        "aws-overview.pdf": "Overview of AWS services",
        "wellarchitected-security-pillar.pdf": "Well-Architected Security Pillar: security best practices",
        "wellarchitected-cost-optimization-pillar.pdf": "Well-Architected Cost Optimization Pillar",
    }
    return sources.get(source_file, f"Available sources: {', '.join(sources.keys())}")


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
        "KMS": "Key Management Service",
    }
    return acronyms.get(acronym.upper().strip(), f"'{acronym}' not in database")


@tool
def list_available_topics() -> str:
    """List AWS topics the chatbot knows about."""
    return "Available: AWS Overview, Security (IAM, encryption), Cost Optimization, Well-Architected Framework"


tools = [search_aws_docs, get_document_source, expand_aws_acronym, list_available_topics]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AWS documentation assistant with access to tools.
- Use search_aws_docs for any AWS technical question.
- Use expand_aws_acronym for acronyms like IAM, EC2, S3.
- Use list_available_topics for questions about what you know.
- Use get_document_source for questions about source files.
- Answer only from retrieved documentation. Do not invent information.
- Always cite the source file and page number.
- Use earlier conversation turns to understand follow-up questions."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)


def _history(messages):
    out = []
    for m in messages[:-1]:
        if m.get("role") == "user":
            out.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "assistant":
            out.append(AIMessage(content=m.get("content", "")))
    return out


def predict(request):
    messages = request.get("messages", []) if isinstance(request, dict) else []
    if not messages:
        answer = "Please ask a question."
    else:
        result = agent_executor.invoke({
            "input": messages[-1].get("content", ""),
            "chat_history": _history(messages),
        })
        answer = result["output"]
    return {
        "choices": [{
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }]
    }


chain = RunnableLambda(predict)
mlflow.models.set_model(chain)
''')


# COMMAND ----------

# Local test
import importlib.util

spec = importlib.util.spec_from_file_location("agent", "/tmp/rag_agent/agent.py")
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)

test = agent_module.chain.invoke({"messages": [{"role": "user", "content": "What is IAM?"}]})
print(test["choices"][0]["message"]["content"])

# COMMAND ----------

# Log and register
import importlib.metadata as md
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.models.rag_signatures import ChatCompletionRequest, ChatCompletionResponse
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint

PACKAGES = [
    "databricks-langchain", "databricks-vectorsearch", "databricks-sdk",
    "langchain", "langchain-core", "langchain-community", "langchain-classic",
    "pydantic", "mlflow",
]
pinned = [f"{p}=={md.version(p)}" for p in PACKAGES]
print("\n".join(pinned))

resources = [
    DatabricksVectorSearchIndex(index_name=VECTOR_INDEX),
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=EMBEDDING_ENDPOINT),
]

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="aws_rag_agent"):
    logged = mlflow.langchain.log_model(
        lc_model="/tmp/rag_agent/agent.py",
        name="agent",
        registered_model_name=MODEL_NAME,
        signature=ModelSignature(inputs=ChatCompletionRequest(), outputs=ChatCompletionResponse()),
        input_example={"messages": [{"role": "user", "content": "What is AWS IAM?"}]},
        pip_requirements=pinned,
        resources=resources,
    )

MODEL_VERSION = str(logged.registered_model_version)
print(f"Registered {MODEL_NAME} version {MODEL_VERSION}")

# COMMAND ----------

# Create or update endpoint
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

entity = ServedEntityInput(
    name="aws_rag_agent",
    entity_name=MODEL_NAME,
    entity_version=MODEL_VERSION,
    workload_size="Small",
    scale_to_zero_enabled=True,
)

existing = [e.name for e in w.serving_endpoints.list()]
if SERVING_ENDPOINT in existing:
    w.serving_endpoints.update_config(name=SERVING_ENDPOINT, served_entities=[entity])
    print(f"Updating {SERVING_ENDPOINT} to version {MODEL_VERSION}")
else:
    w.serving_endpoints.create(name=SERVING_ENDPOINT, config=EndpointCoreConfigInput(name=SERVING_ENDPOINT, served_entities=[entity]))
    print(f"Creating {SERVING_ENDPOINT} with version {MODEL_VERSION}")

# COMMAND ----------

# Wait for endpoint
import time

for _ in range(60):
    st = w.serving_endpoints.get(name=SERVING_ENDPOINT).state
    ready, update = str(st.ready), str(st.config_update)
    print("ready:", ready, "| update:", update)
    if "UPDATE_FAILED" in update:
        print("Build failed, check build logs")
        break
    if ready.endswith("READY") and "NOT" not in ready and "IN_PROGRESS" not in update:
        print("Endpoint ready")
        break
    time.sleep(30)

# COMMAND ----------

# Query endpoint
import json
import requests

headers = w.config.authenticate()
headers["Content-Type"] = "application/json"
URL = f"{w.config.host.rstrip('/')}/serving-endpoints/{SERVING_ENDPOINT}/invocations"


def ask(messages):
    r = requests.post(URL, headers=headers, data=json.dumps({"messages": messages}), timeout=120)
    if r.status_code != 200:
        return f"Error {r.status_code}: {r.text[:1000]}"
    d = r.json()
    if isinstance(d, dict) and "choices" in d:
        return d["choices"][0]["message"]["content"]
    if isinstance(d, list) and d and "choices" in d[0]:
        return d[0]["choices"][0]["message"]["content"]
    if isinstance(d, dict) and "predictions" in d:
        p = d["predictions"]
        p = p[0] if isinstance(p, list) else p
        return p["choices"][0]["message"]["content"]
    return json.dumps(d, indent=2)


history = [{"role": "user", "content": "What is IAM?"}]
a1 = ask(history)
print("Q1: What is IAM?\nA1:", a1, "\n")

history += [{"role": "assistant", "content": a1},
            {"role": "user", "content": "How do I use it to secure my account?"}]
print("Q2: How do I use it to secure my account?\nA2:", ask(history))

# COMMAND ----------

# Latency
import statistics

questions = [
    "What are AWS security best practices?",
    "How can I optimize AWS costs?",
    "What is the AWS Well-Architected Framework?",
    "What does KMS stand for?",
    "How should I protect data at rest?",
    "What is the principle of least privilege?",
    "How do I reduce EC2 costs?",
    "What topics do you know about?",
    "What is in wellarchitected-security-pillar.pdf?",
    "How do I enable MFA?",
]

latencies = []
for q in questions:
    start = time.time()
    ask([{"role": "user", "content": q}])
    latencies.append(time.time() - start)
    print(f"{latencies[-1]:.2f}s  {q}")

print(f"\nMedian latency: {statistics.median(latencies):.2f}s over {len(latencies)} queries")