import os
import pickle
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def load_local(vectorstore_dir: str, embed_model: HuggingFaceEmbeddings) -> tuple[Optional[FAISS], Optional[List[Document]]]:
    """
    Load the vectorstore and documents from disk.
    Args:
        vectorstore_dir: The directory to load the vectorstore from.
        embed_model: The embedding model to use.
    Returns:
        vector_store: The vectorstore.
    """
    from langchain_community.vectorstores import FAISS

    if not os.path.isdir(vectorstore_dir):
        print(f"Vectorstore directory not found at {vectorstore_dir}. Creating a new one.")
        os.makedirs(vectorstore_dir, exist_ok=True)
        
    try:
        vector_store = FAISS.load_local(vectorstore_dir, embed_model, allow_dangerous_deserialization=True)
        
        docs_path = os.path.join(vectorstore_dir, "docs.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                docs = pickle.load(f)
        else:
            docs = None 
            print("Warning: docs.pkl not found. BM25 search will not be available.")

        print(f"Successfully loaded RAG state from {vectorstore_dir}")
        return vector_store, docs
    except Exception as e:
        print(f"Could not load from {vectorstore_dir}. It might be empty or corrupted. Error: {e}")
        return None, None

def save_local(vectorstore_dir: str, vectorstore: FAISS, docs: Optional[List[Document]]) -> None:
    """
    Save the vectorstore and documents to disk.
    Args:
        vectorstore_dir: The directory to save the vectorstore to.
        vectorstore: The vectorstore to save.
        docs: The documents to save.
    """
    if vectorstore is None:
        raise ValueError("Nothing to save.")
    if docs is None:
        print("Warning: No documents to save. BM25 search will not be available.")
    
    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    
    if docs is not None:
        with open(os.path.join(vectorstore_dir, "docs.pkl"), "wb") as f:
            pickle.dump(docs, f)

    print(f"Successfully saved RAG state to {vectorstore_dir}")

def load_qa_dataset(qa_dataset_path: str) -> tuple[List[str], List[str], List[str], List[str]]:
    """
    Load the QA dataset. (jsonl)
    Args:
        qa_dataset_path: The path to the QA dataset.
    Returns:
        Tuple: (ids, questions, options, answers)\\
        ids: The ids of the questions\\
        questions: The questions\\
        options: The options for each question\\
        answers: The answers for each question.
    """
    import json
    if not os.path.exists(qa_dataset_path):
        raise FileNotFoundError(f"Error: File not found at {qa_dataset_path}")
    
    with open(qa_dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    questions = [item["question"] for item in data]
    try:
        options = [
            (f"A. {item['A']} \n" if item['A'] not in [" ", "", None] else "") +
            (f"B. {item['B']} \n" if item['B'] not in [" ", "", None] else "") +
            (f"C. {item['C']} \n" if item['C'] not in [" ", "", None] else "") +
            (f"D. {item['D']} \n" if item['D'] not in [" ", "", None] else "") +
            (f"E. {item['E']} \n" if item['E'] not in [" ", "", None] else "")
            for item in data]
    except KeyError:
        options = [" " for item in data]
    answers = [item["answer"] for item in data]
    uuids = [item["uuid"] for item in data]
    return uuids, questions, options, answers

def load_prepared_retrieve_docs(prepared_retrieve_docs_path: str) -> List[List[Document]]:
    """
    Load the prepared retrieve docs from a file.
    Args:
        prepared_retrieve_docs_path: The path to the prepared retrieve docs.
    Returns:
        A list of lists of documents.
    """
    return safe_load_langchain_docs(prepared_retrieve_docs_path)

def paralelize(func, max_workers: int = 4, **kwargs) -> List:
    """
    Parallelizes a function call over multiple keyword argument iterables.

    Args:
        func: The function to execute in parallel.
        max_workers: The maximum number of threads to use.
        **kwargs: Keyword arguments where each value is an iterable (e.g., a list).
                  All iterables must be of the same length.
                  The keyword names do not matter, but their order does.
    Returns:
        A list of the results of the function calls.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    if not kwargs:
        return []

    arg_lists = list(kwargs.values())
    if len(set(len(lst) for lst in arg_lists)) > 1:
        raise ValueError("All iterable arguments must have the same length.")
        
    total_items = len(arg_lists[0])
    iterable = zip(*arg_lists)
    unpacker_func = lambda args_tuple: func(*args_tuple)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(unpacker_func, iterable), total=total_items))
    return results

def safe_save_langchain_docs(documents: List[List[Document]], filepath: str):
    """
    Converts LangChain Document objects into a serializable list of dictionaries
    and saves them to a file using pickle.

    Args:
        documents (List[List[Document]]): The nested list of LangChain Documents.
        filepath (str): The path to the file where the data will be saved.
    """
    serializable_data = []
    print(f"Preparing to save {len(documents)} lists of documents...")
    
    # Convert each Document object into a dictionary
    for doc_list in documents:
        serializable_doc_list = []
        for doc in doc_list:
            serializable_doc_list.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            })
        serializable_data.append(serializable_doc_list)

    print(f"Conversion complete. Saving to {filepath}...")
    try:
        # Use 'with' to ensure the file is closed properly, even if errors occur
        with open(filepath, "wb") as f:
            pickle.dump(serializable_data, f)
        print("File saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

def safe_load_langchain_docs(filepath: str) -> List[List[Document]]:
    """
    Loads data from a pickle file and reconstructs the LangChain Document objects.

    Args:
        filepath (str): The path to the file to load.

    Returns:
        List[List[Document]]: The reconstructed nested list of LangChain Documents.
    """
    reconstructed_documents = []
    
    print(f"Loading data from {filepath}...")
    try:
        with open(filepath, "rb") as f:
            loaded_data = pickle.load(f)
        print("File loaded successfully. Reconstructing Document objects...")

        # Reconstruct the Document objects from the dictionaries
        for doc_list_data in loaded_data:
            reconstructed_doc_list = []
            for doc_data in doc_list_data:
                reconstructed_doc_list.append(
                    Document(
                        page_content=doc_data["page_content"],
                        metadata=doc_data["metadata"]
                    )
                )
            reconstructed_documents.append(reconstructed_doc_list)
        
        print("Document objects reconstructed successfully.")
        return reconstructed_documents

    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return []
    except EOFError:
        print(f"Error: The file at {filepath} is corrupted or incomplete (EOFError).")
        print("Please re-run the script that generates this file.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        return []