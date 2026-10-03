# Databricks notebook source
# Install required libraries for PDF loading and text chunking
%pip install --quiet pypdf langchain langchain-community langchain-text-splitters transformers
dbutils.library.restartPython()

# COMMAND ----------

# Load AWS documentation PDFs from Databricks volume
import os
from langchain_community.document_loaders import PyPDFLoader

VOLUME_PATH = "/Volumes/workspace/default/documents"

pdf_files = [f for f in os.listdir(VOLUME_PATH) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files in volume.")

all_documents = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(os.path.join(VOLUME_PATH, pdf_file))
    pages = loader.load()
    for page in pages:
        page.metadata["source_file"] = pdf_file
    all_documents.extend(pages)
    print(f"Loaded {pdf_file}: {len(pages)} pages")

print(f"\nTotal pages loaded: {len(all_documents)}")

# COMMAND ----------

# Split loaded PDF pages into smaller text chunks for embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(all_documents)

print(f"Total chunks created: {len(chunks)}")
print(f"Average chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} characters")

# Preview a sample chunk
sample = chunks[0]
print("\nSample chunk metadata:", sample.metadata)
print("Sample chunk content:", sample.page_content[:300])

# Chunk distribution across documents
from collections import Counter
distribution = Counter(c.metadata.get("source_file") for c in chunks)
print("\nChunks per document:")
for source, count in distribution.items():
    print(f"  {source}: {count}")

# COMMAND ----------

# Persist chunks to a Delta table in Unity Catalog
from pyspark.sql import Row

CATALOG = "workspace"
SCHEMA = "default"
TABLE = "pdf_chunks"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Build rows with a unique id, chunk text, and metadata fields
rows = []
for idx, chunk in enumerate(chunks):
    rows.append(Row(
        id=idx,
        chunk_text=chunk.page_content,
        source_file=chunk.metadata.get("source_file"),
        page=int(chunk.metadata.get("page", 0))
    ))

df = spark.createDataFrame(rows)

# Overwrite existing table on each run to keep the pipeline idempotent
(df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_TABLE_NAME))

# Enable Change Data Feed - required by Databricks Vector Search
spark.sql(f"ALTER TABLE {FULL_TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Wrote {df.count()} chunks to {FULL_TABLE_NAME}")
spark.table(FULL_TABLE_NAME).show(3, truncate=80)