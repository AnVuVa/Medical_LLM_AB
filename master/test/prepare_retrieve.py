import argparse
import os

from ..rag_pipeline import get_embeddings, vretrieve
from ..utils import load_local, load_qa_dataset, safe_save_langchain_docs

def main(args):
    embed_model = get_embeddings(args.embed_model_name, show_progress=False)
    vectorstore, docs = load_local(args.vectorstore_dir, embed_model)

    ids, questions, options, answers = load_qa_dataset(args.qa_data_path)
    
    rag_queries = [f"Question: {questions[i]}\n{options[i]}" for i in range(len(questions))]
    if (args.rag_queries_path is not None) and os.path.exists(args.rag_queries_path):
        import json
        with open(args.rag_queries_path, "r", encoding="utf-8") as f:
            rag_queries = [json.loads(line)["query"] for line in f]

    from tqdm import tqdm
    retrieve_results = [vretrieve(rag_queries[i], vectorstore, docs, args.retriever_k, args.metric, args.threshold) for i in tqdm(range(len(rag_queries)), desc="Retrieving documents")]

    safe_save_langchain_docs(retrieve_results, args.prepared_retrieve_docs_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset params
    parser.add_argument("--qa_data_path", type=str, default="dataset/QA Data/MedMCQA/translated_hard_questions.jsonl")

    # Vectorstore params
    parser.add_argument("--vectorstore_dir", type=str, default="notebook/An/master/knowledge/vectorstore_full")
    parser.add_argument("--prepared_retrieve_docs_path", type=str, default="dataset/QA Data/MedMCQA/prepared_retrieve_docs_full.pkl")
    parser.add_argument("--rag_queries_path", type=str, default=None)

    # Model params
    parser.add_argument("--embed_model_name", type=str, default="alibaba-nlp/gte-multilingual-base")

    # Vectorstore retriever params
    parser.add_argument("--vectorstore", type=str, choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--metric", type=str, choices=["cosine", "mmr", "bm25"], default="mmr")
    parser.add_argument("--retriever_k", type=int, default=20, help="Number of documents to retrieve")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for cosine similarity")
    parser.add_argument("--reranker_model_name", type=str, default=None)
    parser.add_argument("--reranker_k", type=int, default=50, help="Number of documents to rerank")

    args = parser.parse_args()
    print(args)

    main(args)