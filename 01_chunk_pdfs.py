# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # PDF ingestion and chunking
# MAGIC
# MAGIC Source PDFs: `/Volumes/workspace/default/documents`  
# MAGIC Output table: `workspace.default.pdf_chunks`

# COMMAND ----------

# MAGIC %pip install --quiet pypdf langchain langchain-community langchain-text-splitters
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Volume for the source PDFs
CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "documents"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.pdf_chunks"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Volume ready: {VOLUME_PATH}")

# COMMAND ----------

# Load PDFs
import os
from langchain_community.document_loaders import PyPDFLoader

pdf_files = sorted(f for f in os.listdir(VOLUME_PATH) if f.endswith(".pdf"))
if not pdf_files:
    raise FileNotFoundError(f"No PDFs found in {VOLUME_PATH}")

all_documents = []
for pdf_file in pdf_files:
    pages = PyPDFLoader(os.path.join(VOLUME_PATH, pdf_file)).load()
    for page in pages:
        page.metadata["source_file"] = pdf_file
    all_documents.extend(pages)
    print(f"Loaded {pdf_file}: {len(pages)} pages")

print(f"\nTotal pages loaded: {len(all_documents)}")

# COMMAND ----------

# Chunking
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(all_documents)

print(f"Total chunks created: {len(chunks)}")
print(f"Average chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

print("\nChunks per document:")
for source, count in Counter(c.metadata["source_file"] for c in chunks).items():
    print(f"  {source}: {count}")

# COMMAND ----------

# Write chunks to Delta
from pyspark.sql import Row

rows = [
    Row(
        id=idx,
        chunk_text=chunk.page_content,
        source_file=chunk.metadata["source_file"],
        # PyPDFLoader pages are 0-based
        page=int(chunk.metadata.get("page", 0)) + 1,
    )
    for idx, chunk in enumerate(chunks)
]

df = spark.createDataFrame(rows)

(df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_TABLE_NAME))

# Required for Delta Sync index
spark.sql(f"ALTER TABLE {FULL_TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Wrote {df.count()} chunks to {FULL_TABLE_NAME}")
display(spark.table(FULL_TABLE_NAME).limit(3))