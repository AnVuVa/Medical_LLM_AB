import argparse
import os
from typing import List

from ..rag_pipeline import get_embeddings, load_data
from ..utils import load_local, save_local

def main(args):
    print(f"Log: {args}")

    if args.clear_vectorstore:
        import shutil
        if os.path.isdir(args.vectorstore_dir):
            shutil.rmtree(args.vectorstore_dir)

    embed_model = get_embeddings(args.embed_model_name)
    vectorstore, docs = load_local(args.vectorstore_dir, embed_model)

    new_docs = []
    for data_path in args.data_paths:
        new_docs.extend(load_data(data_path, args.file_type))
    print(f"Got {len(new_docs)} documents.")

    if args.chunk_method == "recursive":
        from ..rag_pipeline import recursive_chunking
        new_docs = recursive_chunking(new_docs, args.chunk_size, args.chunk_overlap)
    elif args.chunk_method == "markdown":
        from ..rag_pipeline import markdown_chunking
        new_docs = markdown_chunking(new_docs, args.chunk_size, args.chunk_overlap)
    print(f"Got {len(new_docs)} chunks.")

    from langchain_community.vectorstores import FAISS
    if vectorstore is None:
        vectorstore = FAISS.from_documents(new_docs, embed_model)
        docs = new_docs
        print(f"Successfully consumed {len(new_docs)} documents.")
    else:
        docs.extend(new_docs)
        vectorstore.add_documents(new_docs)

    save_local(args.vectorstore_dir, vectorstore, docs)

    import json
    with open(os.path.join(args.vectorstore_dir, "config.json"), "a") as f:
        json.dump(vars(args), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    data_paths = [
        'dataset/RAG_Data/wiki_vi',
        'dataset/RAG_Data/youmed',
        'dataset/RAG_Data/mimic_ex_report',
        'dataset/RAG_Data/Download sach y/OCR',
    ]

    # Dataset params
    parser.add_argument("--data_paths", type=List[str], required=False, default=data_paths)
    parser.add_argument("--vectorstore_dir", type=str, required=False, default="notebook/An/master/knowledge/vectorstore_full")
    parser.add_argument("--file_type", type=str, choices=["pdf", "txt"], default="txt")
 
    # Model params
    parser.add_argument("--embed_model_name", type=str, default="alibaba-nlp/gte-multilingual-base")

    # Index params
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--chunk_overlap", type=int, default=512)
    parser.add_argument("--chunk_method", type=str, choices=["recursive", "markdown"], default="markdown")

    # Vectorstore params
    parser.add_argument("--vectorstore", type=str, choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--clear_vectorstore", action="store_true", default=True)


    args = parser.parse_args()

    main(args)