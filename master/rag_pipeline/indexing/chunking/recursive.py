from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def __split_1_document__(document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    text_content = document.page_content
    text_chunks = text_splitter.split_text(text_content)
    split_documents = []
    
    for i, chunk in enumerate(text_chunks):
        new_metadata = document.metadata.copy()
        
        # new_metadata['chunk_number'] = i + 1
        
        new_doc = Document(page_content=chunk, metadata=new_metadata)
        split_documents.append(new_doc)
        
    return split_documents


def split_document(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    split_documents = []
    for doc in documents:
        split_documents.extend(__split_1_document__(doc, chunk_size, chunk_overlap))
    return split_documents