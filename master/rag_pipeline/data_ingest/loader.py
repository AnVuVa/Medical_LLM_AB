import os
from typing import List
from langchain.schema import Document

def load_data(data_path: str, file_type: str) -> List[Document]:
    """
    Load knowledge data from a specified path and file type.
    Args:
        data_path: The path to the data.
        file_type: The type of the data.
    Returns:
        A list of documents.
    """
    if file_type == "pdf":
        raise NotImplementedError("PDF loading is not yet implemented.")
    elif file_type == "txt":
        return _load_txt(data_path)

def _load_txt(data_path: str) -> List[Document]:
    splits = []

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Error: Directory not found at {data_path}")

    for file_name in os.listdir(data_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_path, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata = {"source": file_name}
                doc = Document(page_content=content, metadata=metadata)
                
                splits.append(doc)

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                
    return splits
