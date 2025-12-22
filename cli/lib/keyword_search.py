import os
import pickle
import string
import math
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_tf(self, term: str, doc_id: int) -> int:
        if ' ' in term:
            raise ValueError("Term must be a single token")
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)
    
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(term, doc_id)
        avg_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths.get(doc_id, 0) / avg_length) if avg_length > 0 else 1
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm) if tf > 0 else 0.0
        return bm25_tf
    
    def bm25(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term, k1, b)
        return idf * bm25_tf
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT, k1: float = BM25_K1, b: float = BM25_B) -> list[dict]:
        query_tokens = tokenize_text(query)
        scores = defaultdict(float)
        for token in query_tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] += self.bm25(doc_id, token, k1, b)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        results = [self.docmap[doc_id] for doc_id, _ in ranked_docs]
        for i, (doc_id, score) in enumerate(ranked_docs):
            results[i]['score'] = score
        return results
    
    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            raise RuntimeError("Inverted index not found. Please build the index first.")
        
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        avg_length = total_length / len(self.doc_lengths) if self.doc_lengths else 0
        return avg_length
       

def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT, k1: float = BM25_K1, b: float = BM25_B) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = idx.bm25_search(query, limit, k1, b)
    return results


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    token = tokens[0]
    bm25_tf = idx.get_bm25_tf(doc_id, token, k1, b)
    return bm25_tf
        

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    token = tokens[0]
    idf = idx.get_bm25_idf(token)
    return idf


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(term, doc_id)
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    token = tokens[0]
    N = len(idx.docmap)
    df = len(idx.get_documents(token))
    idf = math.log((N + 1) / (df + 1))
    tfidf = tf * idf
    return tfidf

        
def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(term, doc_id)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    token = tokens[0]
    N = len(idx.docmap)
    df = len(idx.get_documents(token))
    idf = math.log((N + 1) / (df + 1))
    return idf


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    results = []
    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for doc_id in doc_ids:
            movies = idx.docmap[doc_id]
            results.append(movies)
            if len(results) >= limit:
                break

    return results


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
