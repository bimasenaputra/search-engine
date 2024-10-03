# Search Engine

![CLI text search engine](https://drive.google.com/uc?export=view&id=1uS86lHJ7QAAUJtxjMHsGaQyO3lgVzpX5)

CLI text search engine made from scratch using Python built on top of well-known inverted index architecture.

## 🎨 Features
- `Boolean retrieval`: retrieve
- `TF-IDF ranked retrieval`: retrieve_tfidf
- `BM25 ranked retrieval`: retrieve_bm25
- `Fast WAND TF-IDF ranked retrieval`: retrieve_wand_tfidf
- `Learning-to-Rank LambdaMART reranking`: retrieve_letor
- `Dense Passage Retriever(DPR) reranking`: retrieve_dpr
- `Posting compression`
- `Evaluation metrics`

### 🐾 Installation and Example
First, clone this repository
```
git clone https://github.com/bimasenaputra/search-engine.git
```

Next, install all necessary python packages.

Finally, try running some example queries using BM25 ranked retrieval by running `search.py`.

```
python search.py
```

You can change the ranking strategy by editing search.py.

**Important for DPR reranking**

Before using DPR reranking, make sure you have internet connection to download the pre-trained model.

If you want to use or finetune your own model, you can utilize squad_to_dpr.py which convert your dataset from SQuAD format to DPR format.

Alternatively, you can train the model with the provided bioASQ dataset by running `dpr.py`, then load the trained model.
### 📋 Evaluation
`Under construction`

Currently it is written in bonus.txt & evaluasi.txt.

To evaluate ranking yourself, run `experiment.py`.
### Future Works
- Implement FAISS
