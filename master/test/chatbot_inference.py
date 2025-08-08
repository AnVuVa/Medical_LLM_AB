from rag_pipeline import get_embeddings, vretrieve, rerank
from utils import load_local

import argparse

def inference():
    embed_model = get_embeddings(args.embed_model_name)
    vectorstore, docs = load_local(args.vectorstore_dir, embed_model)
    retrieve_results = vretrieve(args.query, vectorstore, docs, args.retriever_k, args.metric, args.threshold)
    
    retrieve_results = rerank(retrieve_results)

    print(retrieve_results)

def conversation():
    while True:
        query = input("User: ")
        if query == "exit":
            break
        inference(query)

if __name__ == '__main__':
    conversation()