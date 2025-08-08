from langchain_huggingface import HuggingFaceEmbeddings

import torch

_model_cache = {}

def get_embeddings(model_name: str, show_progress: bool = True) -> HuggingFaceEmbeddings:
    """
    Get the embeddings model. Cache available.
    Args:
        model_name: The name of the model.
    Returns:
        The embeddings model.
    """
    if model_name not in _model_cache:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            show_progress=show_progress,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'trust_remote_code':True},
            encode_kwargs={'batch_size': 15}
        )
        _model_cache[model_name] = embeddings
    return _model_cache[model_name]