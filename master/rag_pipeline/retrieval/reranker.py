import os
import pickle
from typing import List

from langchain.schema import Document

def rerank(docs: List[Document]) -> List[Document]:
    return docs