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

def build_term_documents_mat(df: pd.DataFrame, boolean_matrix : bool = False) -> np.array:

    docs = df["clean_text"]

    vocab , term_freq = create_vocab(docs)

    sorted_vocab = sorted(vocab)
    indexed_vocab = {string: idx for idx, string in enumerate(sorted_vocab)}

    term_document_matrix = np.zeros((len(vocab), len(docs)))

    for doc_idx, tf_doc in enumerate(term_freq):
        for term, freq in tf_doc.items():

            if boolean_matrix:
                freq = 1

            term_document_matrix[indexed_vocab[term], doc_idx] = freq

    return term_document_matrix.T, np.array(sorted_vocab)

# Example usage
if __name__ == '__main__':
    df = pd.read_parquet("./data/test/test.parquet")
    td_matrix = build_term_documents_mat(df)
    print(td_matrix)