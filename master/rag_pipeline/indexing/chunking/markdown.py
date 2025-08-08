from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def __split_1_document__(document: Document, chunk_size: int, chunk_overlap: int) -> List[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
        return_each_line=False
    )
    
    md_header_splits = markdown_splitter.split_text(document.page_content)

    for doc in md_header_splits:
        doc.metadata.update(document.metadata)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    
    # Iterate through the final chunks to prepend metadata to the page_content
    for i, doc in enumerate(final_splits):
        header_lines = []
        source_line = f"-- source: {doc.metadata.get('source', 'N/A')}"
        
        if 'Header 1' in doc.metadata:
            header_lines.append(doc.metadata['Header 1'])
        if 'Header 2' in doc.metadata:
            header_lines.append(doc.metadata['Header 2'])
        if 'Header 3' in doc.metadata:
            header_lines.append(doc.metadata['Header 3'])
            
        header_content = "\n".join(header_lines)
        chunk_header = f"Chunk {i+1}:"
        
        # Combine everything into the new page content
        original_content = doc.page_content
        doc.page_content = f"{source_line}\n{header_content}\n{chunk_header}\n{original_content}"

    return final_splits

def split_document(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    split_documents = []
    for doc in documents:
        split_documents.extend(__split_1_document__(doc, chunk_size, chunk_overlap))
    return split_documents