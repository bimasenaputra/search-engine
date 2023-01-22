# Search Engine

CLI text search engine made from scratch using Python built on top of well-known inverted index architecture.

## 🎨 Features
- `Boolean retrieval`: retrieve
- `TF-IDF ranked retrieval`: retrieve_tfidf
- `BM25 ranked retrieval`: retrieve_bm25
- `Fast WAND TF-IDF ranked retrieval`: retrieve_wand_tfidf
- `Learning-to-Rank LambdaMART reranking`: retrieve_letor
- `Dense Passage Retriever reranking`: retrieve_dpr
- `Posting compression`
- `Evaluation metrics`

### 🐾 Installation and Example
First, clone this repository
```
git clone https://github.com/bimasenaputra/search-engine.git
```

Next, install python packages
```
pip install -r /path/to/requirements.txt
```

Try querying example query using BM25 ranked retrieval by running `search.py`.
You can change the ranking strategy by editing search.py.

### Evaluation
`Under construction`
