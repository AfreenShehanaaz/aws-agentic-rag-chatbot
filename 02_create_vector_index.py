# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Vector search index
# MAGIC
# MAGIC Delta Sync index on `workspace.default.pdf_chunks`, embedded with Qwen3.

# COMMAND ----------

# MAGIC %pip install --quiet databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

VS_ENDPOINT = "rag_endpoint"
SOURCE_TABLE = "workspace.default.pdf_chunks"
INDEX_NAME = "workspace.default.pdf_chunks_index"
EMBEDDING_ENDPOINT = "databricks-qwen3-embedding-0-6b"

# COMMAND ----------

# Vector Search endpoint
import time
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
print("Existing Vector Search endpoints:", existing)

if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")
    print(f"Creating {VS_ENDPOINT}...")

for _ in range(60):
    state = vsc.get_endpoint(VS_ENDPOINT).get("endpoint_status", {}).get("state", "UNKNOWN")
    print(time.strftime("%H:%M:%S"), "endpoint state:", state)
    if state == "ONLINE":
        break
    time.sleep(20)

# COMMAND ----------

# Create or sync index
existing_indexes = [
    i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])
]

if INDEX_NAME in existing_indexes:
    index = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
    index.sync()
    print(f"Index exists. Sync started for {INDEX_NAME}")
else:
    index = vsc.create_delta_sync_index(
        endpoint_name=VS_ENDPOINT,
        source_table_name=SOURCE_TABLE,
        index_name=INDEX_NAME,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
    print(f"Creating index {INDEX_NAME}...")

# COMMAND ----------

# Wait for index
last_line = None
for _ in range(90):
    status = index.describe().get("status", {})
    ready = status.get("ready", False)
    line = f"state: {status.get('detailed_state')} | rows indexed: {status.get('indexed_row_count')}"
    if line != last_line:
        print(line)
        last_line = line
    if ready:
        print("Index ready")
        break
    time.sleep(20)

# COMMAND ----------

# Hybrid search check
result = index.similarity_search(
    query_text="What are AWS security best practices?",
    columns=["source_file", "page", "chunk_text"],
    num_results=3,
    query_type="HYBRID",
)

for source_file, page, text, score in result["result"]["data_array"]:
    print(f"[{source_file} p.{int(page)}] score={score:.3f}")
    print(text[:200], "\n")