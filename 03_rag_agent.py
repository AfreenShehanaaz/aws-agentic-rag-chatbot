# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # RAG chain and tool-calling agent

# COMMAND ----------

# MAGIC %pip install --quiet databricks-langchain databricks-vectorsearch databricks-sdk langchain langchain-core langchain-community langchain-classic mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Config
import mlflow
from databricks.sdk import WorkspaceClient

VECTOR_ENDPOINT = "rag_endpoint"
VECTOR_INDEX = "workspace.default.pdf_chunks_index"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
TOP_K = 3

user = WorkspaceClient().current_user.me().user_name
mlflow.set_experiment(f"/Users/{user}/aws_rag_experiments")

mlflow.langchain.autolog()

# COMMAND ----------

# Retriever (hybrid search)
from databricks_langchain import DatabricksVectorSearch, ChatDatabricks

vector_store = DatabricksVectorSearch(
    endpoint=VECTOR_ENDPOINT,
    index_name=VECTOR_INDEX,
    columns=["source_file", "page"],
)
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K, "query_type": "HYBRID"})
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1, max_tokens=800)

for i, d in enumerate(retriever.invoke("What are AWS security best practices?"), 1):
    print(f"[{i}] {d.metadata.get('source_file')} p.{int(d.metadata.get('page', 0))}")
    print(d.page_content[:200], "\n")

# COMMAND ----------

# RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag_prompt = ChatPromptTemplate.from_template(
    "You are an AWS documentation assistant. Answer the question using only "
    "the context below. If the context does not contain the answer, say you "
    "do not have that information. Cite the source file and page at the end.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source_file')} p.{int(d.metadata.get('page', 0))}]\n{d.page_content}"
        for d in docs
    )


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

for q in ["What are AWS security best practices?",
          "How can I optimize AWS costs?",
          "What is the AWS Well-Architected Framework?"]:
    print(f"Q: {q}\nA: {rag_chain.invoke(q)}\n" + "-" * 80)

# COMMAND ----------

# Tools
from langchain_core.tools import tool


@tool
def search_aws_docs(query: str) -> str:
    """Search AWS documentation for information about a specific topic."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant AWS documentation was found."
    return format_docs(docs)


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

# COMMAND ----------

# Agent with memory
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

agent_prompt = ChatPromptTemplate.from_messages([
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

agent = create_tool_calling_agent(llm, tools, agent_prompt)
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
    history_messages_key="chat_history",
)


# COMMAND ----------

# Multi-turn test
session = {"configurable": {"session_id": "test-1"}}

for q in ["What is IAM?", "How do I use it to secure my account?", "What tools help with this?"]:
    print("=" * 70 + f"\nQ: {q}\n" + "=" * 70)
    print("A:", agent_with_memory.invoke({"input": q}, config=session)["output"], "\n")
