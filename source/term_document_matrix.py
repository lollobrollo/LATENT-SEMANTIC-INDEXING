"""
    This code is developed as final project for the Information Retrieval course (2024 - 2025).
    Authors : Bortolussi L., Bredariol F., Tonet L.
"""

import pandas as pd
import numpy as np
from collections import Counter

def create_vocab(docs: pd.DataFrame) -> list:
    """
    Creates a vocabulary list and computes term frequencies for each document.

    Parameters:
    - docs (pd.DataFrame): A pandas Series or list-like object containing documents as strings.

    Returns:
    - list: A sorted list of unique terms (vocabulary).
    - list of dictionaries: A list where each element is a dictionary term:frequency for every document.
    """

    vocab = []
    term_frequencies = []

    # iterate every document in the list
    for doc in docs:
        # get a list of single lemmas
        temp = doc.split(" ")
        # get a dictionary {term:frequency}
        doc_tf = Counter(temp)

        term_frequencies.append(doc_tf)

        #add all (unique) terms in the vocabulary
        vocab = vocab + list(set(temp))

    return list(set(vocab)), term_frequencies

def build_term_documents_mat(df: pd.DataFrame, boolean_matrix: bool = False) -> np.array:
    """
    Builds a term-document matrix from preprocessed text in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'clean_text' column with preprocessed text (space-separated tokens).
    - boolean (bool): True for binary term presence instead of frequency.

    Returns:
    - np.array: A 2D NumPy array where rows are terms and columns are documents.
                Each entry represents the frequency of a term in a document.
    """
        
    docs = df["clean_text"]

    vocab , term_freq = create_vocab(docs)

    sorted_vocab = sorted(vocab)
    indexed_vocab = {string: idx for idx, string in enumerate(sorted_vocab)}

    term_document_matrix = np.zeros((len(vocab), len(docs)))

    for doc_idx, tf_doc in enumerate(term_freq):
        for term, freq in tf_doc.items():

            if boolean_matrix: freq = 1

            term_document_matrix[indexed_vocab[term], doc_idx] = freq

    return term_document_matrix.T, np.array(sorted_vocab)

# Example usage
if __name__ == '__main__':
    df = pd.read_parquet("./data/test/test.parquet")
    td_matrix = build_term_documents_mat(df)
    print(td_matrix)
    
