"""
    This code is developed as final project for the Information Retrieval course (2024 - 2025).
    Authors : Bortolussi L., Bredariol F., Tonet L.
"""
import pandas as pd
import numpy as np
from collections import Counter

def create_vocab(docs: pd.DataFrame) -> list:
    vocab = []
    term_frequencies = []
    for doc in docs:
        temp = doc.split(" ")
        doc_tf = Counter(temp)
        term_frequencies.append(doc_tf)
        vocab = vocab + list(set(temp))

    return list(set(vocab)), term_frequencies

def build_term_documents_mat(df: pd.DataFrame) -> np.array:

    docs = df["clean_text"]

    vocab , tf = create_vocab(docs)
    
    return