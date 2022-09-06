# -*- coding: utf-8 -*-
import time

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import sys
from text2vec import SentenceModel, cos_sim, semantic_search, BM25
import torch

#################################################################################
# 1. connect to Milvus
connections.connect("default", host="10.112.83.133", port="19530")

has = utility.has_collection("lhrNeurSearch")
print(f"Does collection lhrNeurSearch exist in Milvus: {has}")

#################################################################################
# 2. create collection
# We're going to create a collection with 2 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |   VarChar  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|"embeddings"| FloatVector|     dim=768      |  "float vector with dim 768" |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=50000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)
]

schema = CollectionSchema(fields, "lhrNeurSearch is the simplest demo to introduce the APIs")

#print(fmt.format("Create collection `lhr-TextMatch`"))
hello_milvus = Collection("lhrNeurSearch", schema, consistency_level="Strong")

################################################################################
# # 3. insert data

embedder = SentenceModel()
corpus=[]
with open('dev.csv','r') as fp:
    for line in fp:
        x=line[:-1].split(",")
        corpus.append(x[0])
        corpus.append(x[1])
corpus=list(set(corpus))
corpus_embeddings = embedder.encode(corpus)
norms = np.linalg.norm(corpus_embeddings, axis=1)
corpus_embeddings=corpus_embeddings/np.tile(norms,(768,1)).T

sentences = [corpus,corpus_embeddings]
insert_result = hello_milvus.insert(sentences)

################################################################################
# 4. create index

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 768},
}

hello_milvus.create_index("embeddings", index)

################################################################################
# 5. search, query, and hybrid search
hello_milvus.load()
# -----------------------------------------------------------------------------
# search based on vector similarity
queries=[]
with open('predict.csv','r') as fp:
    for line in fp:
        x=line[:-1].split(",")
        queries.append(x[0])
        queries.append(x[1])
queries=list(set(queries))[5:10]
query_embeddings = embedder.encode(queries)
norms = np.linalg.norm(query_embeddings, axis=1)
query_embeddings=query_embeddings/np.tile(norms,(768,1)).T

search_params = {
    "metric_type": "IP",
    "params": {"nprobe": 5},
}
start_time = time.time()
result = hello_milvus.search(query_embeddings, "embeddings", search_params, limit=5)
end_time = time.time()
cnt=0
for hits in result:
    print("\n\n======================\n\n")
    print("Query:")
    print(queries[cnt])
    cnt=cnt+1
    print("\nTop 5 most similar sentences in database:")
    j=0
    for hit in hits:
        print(hit.id, "(Score: {:.4f})".format(hit.score))
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 7. drop collection
utility.drop_collection("lhrNeurSearch")