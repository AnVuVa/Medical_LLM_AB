import argparse
import os

from ..rag_pipeline import get_embeddings, rerank
from ..utils import load_local

from ..rag_pipeline import vretrieve

def main(args):
    embed_model = get_embeddings(args.embed_model_name)
    vectorstore, docs = load_local(args.vectorstore_dir, embed_model)
    retrieve_results = vretrieve(args.query, vectorstore, docs, args.retriever_k, args.metric, args.threshold)
    
    retrieve_results = rerank(retrieve_results)

    print(retrieve_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, required=False, default="What are the applications of beta blockers in the treatment of hypertension?")

    # Vectorstore params
    parser.add_argument("--vectorstore_dir", type=str, required=False, default="notebook/An/master/knowledge/vectorstore_full")
 
    # Model params
    parser.add_argument("--embed_model_name", type=str, default="alibaba-nlp/gte-multilingual-base")

    # Vectorstore retriever params
    parser.add_argument("--vectorstore", type=str, choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--metric", type=str, choices=["cosine", "mmr", "bm25"], default="cosine")
    parser.add_argument("--retriever_k", type=int, default=4, help="Number of documents to retrieve")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for cosine similarity")
    parser.add_argument("--reranker_model_name", type=str, default=None)
    parser.add_argument("--reranker_k", type=int, default=20, help="Number of documents to rerank")

    args = parser.parse_args()

    main(args)