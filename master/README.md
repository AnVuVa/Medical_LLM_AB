```
python -m notebook.An.master.test.data_ingest 
--data_dir notebook/An/master/data \\
--vectorstore_dir notebook/An/master/knowledge/vectorstore_1 \\
--embed_model_name alibaba-nlp/gte-multilingual-base \\
--chunking_strategy recursive \\
--chunk_size 2048 \\
--chunk_overlap 512 \\
--vectorstore faiss
```

```
python -m notebook.An.master.test.test_retrieve 
--query "Heart definition and heart disease" 
--vectorstore_dir notebook/An/master/knowledge/vectorstore_1 \\
--embed_model_name alibaba-nlp/gte-multilingual-base \\
--retriever_k 4 \\
--metric cosine \\
--threshold 0.5 \\
```