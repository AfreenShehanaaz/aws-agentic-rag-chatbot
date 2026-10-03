# 🤖 AWS Agentic RAG Assistant & Tool-Calling Platform

An enterprise-grade, end-to-end **Agentic RAG (Retrieval-Augmented Generation)** assistant built on **Databricks**, powered by **Meta Llama 3.3 70B** and **LangChain**.

The system ingests 550+ pages of official AWS documentation, indexes chunks using **Databricks Hybrid Vector Search**, and equips a tool-calling AI agent with dynamic reasoning and execution capabilities (vector retrieval, acronym expansion, source metadata inspection).

---

## 📐 System Architecture

1. **Ingestion & Data Processing:** Technical PDFs ingested from Databricks Volumes via `PyPDFLoader` and split using `RecursiveCharacterTextSplitter` (1000 size / 200 overlap).
2. **Delta Lake & Change Data Feed:** Chunks stored in Delta Lake (`workspace.default.pdf_chunks`) with `delta.enableChangeDataFeed = true` for automated index sync.
3. **Hybrid Vector Search:** High-dimensional vector index powered by `databricks-qwen3-embedding-0-6b` via Databricks Vector Search.
4. **Agentic Tool Calling:** Built using `create_tool_calling_agent` and `AgentExecutor` (`max_iterations=5`) with 4 custom `@tool` functions.
5. **Conversational Memory:** Stateful session tracking via `RunnableWithMessageHistory` and `InMemoryChatMessageHistory`.
6. **MLOps & Serverless Serving:** Executions auto-traced with **MLflow**, models registered in **Unity Catalog**, and deployed via **Databricks Model Serving** (Serverless REST Endpoint).

---

## 🛠️ Tech Stack & Models

* **Language:** Python, PySpark
* **LLM Engine:** Meta Llama 3.3 70B (`databricks-meta-llama-3-3-70b-instruct`)
* **Embedding Model:** Qwen 3 Embedding 0.6B (`databricks-qwen3-embedding-0-6b`)
* **Frameworks:** LangChain, MLflow, Delta Lake, Unity Catalog
* **Deployment:** Databricks Model Serving (Serverless REST Endpoint), Databricks Apps

---

## 🔧 Custom Agent Tools (`@tool`)

1. `search_aws_docs(query)`: Executes similarity search on Databricks Vector Search.
2. `expand_aws_acronym(acronym)`: Resolves technical AWS service terms (IAM, EC2, S3, VPC, KMS, RDS, MFA).
3. `get_document_source(source_file)`: Inspects source PDF metadata.
4. `list_available_topics()`: Displays knowledge domain scope.
