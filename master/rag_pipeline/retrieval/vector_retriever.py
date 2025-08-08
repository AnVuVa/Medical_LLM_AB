from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from .reranker import rerank

from typing import List, Any

def retrieve(query: str, vectorstore: FAISS, docs: List[Document] = None, k: int = 4, metric: str = "cosine", threshold: float = 0.5, reranker: Any = None) -> List[Document]:
    """
    Retrieve documents from the vectorstore based on the query and metric.
    Args:
       query: The query to search for.
       metric: The metric to use for retrieval.
       vectorstore: The vectorstore to search in.
       k: The number of documents to retrieve.
       threshold: The threshold for the metric to use for retrieval.
       reranker: The reranker to use for reranking the retrieved documents.
    Returns:
       A list of documents.
    """
    if metric == "cosine":
        docs = vectorstore.similarity_search_with_score(query, k=k)
        docs = [doc for doc, score in docs if score > threshold]
    elif metric == "mmr":
        docs = vectorstore.max_marginal_relevance_search(query, k=k)
    elif metric == "bm25":
        from langchain_community.retrievers import BM25Retriever
        if docs is None:
            raise ValueError("Documents not available. BM25 requires ingested or loaded documents.")
        bm25_retriever = BM25Retriever.from_documents(docs)
        docs = bm25_retriever.get_relevant_documents(query, k=k)
    else:
        raise ValueError(f"Unsupported metric: '{metric}'. Supported metrics are 'similarity', 'mmr', and 'bm25'.")
    
    if (reranker != None):
        return rerank(docs)
    return docs