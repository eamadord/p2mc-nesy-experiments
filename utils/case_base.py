"""
Base de casos del CBR.

Un caso = {"arxiv_id", "abstract", "fields": {name, category, tasks, repo, datasets, metrics}}.
La recuperación es por similitud coseno entre abstracts.

Sustituye a la colección de ChromaDB del experimento anterior por una
implementación en memoria (numpy), determinista y persistible en disco.
Embedder por defecto: all-MiniLM-L6-v2 (el mismo que usa ChromaDB por defecto),
con TF-IDF como alternativa ligera que no necesita descargar nada.
"""
import json
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, kind="minilm"):
        self.kind = kind
        self._model = None
        self._tfidf = None

    def fit(self, texts):
        """Solo necesario para TF-IDF (no supervisado: usa abstracts, nunca etiquetas)."""
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2)
            self._tfidf.fit(texts)
        return self

    def _load_minilm(self):
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            return lambda xs: m.encode(xs, batch_size=64, show_progress_bar=False)
        except ImportError:
            from chromadb.utils import embedding_functions
            ef = embedding_functions.DefaultEmbeddingFunction()  # ONNX all-MiniLM-L6-v2
            return lambda xs: np.array(ef(xs))

    def encode(self, texts):
        texts = [t or "" for t in texts]
        if self.kind == "tfidf":
            if self._tfidf is None:
                raise RuntimeError("Llama a Embedder.fit(textos) antes de usar TF-IDF.")
            x = self._tfidf.transform(texts).toarray().astype(np.float32)
        else:
            if self._model is None:
                self._model = self._load_minilm()
            x = np.asarray(self._model(texts), dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms


class CaseBase:
    def __init__(self, embedder):
        self.embedder = embedder
        self.ids, self.cases = [], []
        self._emb = None

    def __len__(self):
        return len(self.cases)

    def add_many(self, cases, embeddings=None):
        """cases: lista de dicts {arxiv_id, abstract, fields}."""
        if not cases:
            return
        if embeddings is None:
            embeddings = self.embedder.encode([c["abstract"] for c in cases])
        embeddings = np.asarray(embeddings, dtype=np.float32)
        self.ids.extend(c["arxiv_id"] for c in cases)
        self.cases.extend(cases)
        self._emb = embeddings if self._emb is None else np.vstack([self._emb, embeddings])

    def add(self, case, embedding=None):
        self.add_many([case], None if embedding is None else np.asarray(embedding)[None, :])

    def retrieve(self, abstract=None, k=1, exclude_id=None, query_embedding=None):
        """Devuelve [(similitud, caso), ...] ordenados de mayor a menor similitud."""
        if not self.cases:
            return []
        q = query_embedding if query_embedding is not None else self.embedder.encode([abstract])[0]
        sims = self._emb @ np.asarray(q, dtype=np.float32)
        if exclude_id is not None:
            for i, cid in enumerate(self.ids):
                if cid == exclude_id:
                    sims[i] = -np.inf
        order = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.cases[i]) for i in order if np.isfinite(sims[i])]

    # Persistencia (para el pipeline de producción)
    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "cases.jsonl", "w", encoding="utf-8") as f:
            for c in self.cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        if self._emb is not None:
            np.save(d / "embeddings.npy", self._emb)

    def load(self, directory):
        d = Path(directory)
        if not (d / "cases.jsonl").is_file():
            return self
        with open(d / "cases.jsonl", encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]
        emb = np.load(d / "embeddings.npy") if (d / "embeddings.npy").is_file() else None
        self.ids, self.cases, self._emb = [], [], None
        self.add_many(cases, emb)
        return self
