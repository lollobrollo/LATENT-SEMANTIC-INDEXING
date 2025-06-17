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


def build_term_documents_mat(df: pd.DataFrame, metric: str = "freq") -> np.array:
    """
    Builds a term-document matrix from preprocessed text in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'clean_text' column with preprocessed text (space-separated tokens).
    - metric (string): one between {"bool", "freq", "tf-idf"}

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

            if metric == "bool": freq = 1

            term_document_matrix[indexed_vocab[term], doc_idx] = freq
    
    if metric == "tf-idf":
        term_document_matrix = build_tfidf_mat(term_document_matrix)
        
    return term_document_matrix, np.array(sorted_vocab)


def compute_doc_freq(td_matrix: np.array) -> np.array:
    """
    Counts the documents where a term appears

    Parameters:
    - td_matrix (np.array): a precomputed term document matrix. It can be "one-hot encoded" or "term frequencies".

    Returns:
    - np.array: "document frequency" for each term. The idx of terms matches the one of the matrix
    """

    return np.count_nonzero(td_matrix, axis=1)

def compute_idf(td_matrix: np.array, document_freq: np.array) -> np.array:
    """
    Computes the Inverse Document Frequency (IDF) for each term.

    Parameters:
    - td_matrix (np.array): term-document matrix (can contain term frequencies or binary values).
    - document_freq (np.array): array of document frequencies (i.e., number of documents where each term appears).

    Returns:
    - np.array: IDF score for each term. Same order as the rows of td_matrix.
    """

    n_docs = td_matrix.shape[1]
    return np.log10(n_docs / (document_freq + 1))


def build_tfidf_mat(td_matrix: np.array) -> np.array:
    """
    Builds the TF-IDF matrix by combining term frequencies with IDF scores.

    Parameters:
    - td_matrix (np.array): term-document matrix with term frequencies.

    Returns:
    - np.array: TF-IDF matrix with same shape as td_matrix.
    """
    doc_freq = compute_doc_freq(td_matrix)
    idf_array = compute_idf(td_matrix, doc_freq)

    tfidf_matrix = td_matrix.copy()
    for i in range(td_matrix.shape[0]):
        tfidf_matrix[i, :] *= idf_array[i]

    return tfidf_matrix

# Example usage
if __name__ == '__main__':
    df = pd.read_parquet(r"D:\UNI\3° ANNO\DATA VIS & INFORMATION RETRIEVAL\IR Project\LSI\data\test\test.parquet")
    td_matrix = build_term_documents_mat(df)

    print(td_matrix.shape)
    
