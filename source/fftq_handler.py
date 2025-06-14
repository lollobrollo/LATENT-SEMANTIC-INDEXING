"""
    This code is developed as final project for the Information Retrieval course (2024 - 2025).
    Authors : Bortolussi L., Bredariol F., Tonet L.
"""

import spacy
import re
import numpy as np

def preprocess_query_for_lsi(
    query : str,
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stop: bool = True,
    lemmatize: bool = True,
    remove_num: bool = True,
    allowed_pos: list = None
) -> str:
    """
    Performs  preprocessing for our LSI model.

    Parameters:
    - query (str): Input free form text query.
    - lowercase (bool) : If True, convert text to lowercase.
    - remove_punct (bool) : If True, remove punctuation tokens.
    - remove_stop (bool) : If True, remove stopword tokens.
    - lemmatize (bool) : If True, lemmatize tokens to their base form.
    - remove_num (bool) : If True, remove number-like tokens.
    - allowed_pos (list) : A list of Part-of-Speech tags to keep.
                          Example: ['NOUN', 'PROPN', 'ADJ', 'VERB'].
                          If None, all tokens are kept.

    Returns:
    - formatted string (str): correctly formatted query.
    """
    if allowed_pos is None:
        allowed_pos = ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'] # Default meaningful POS

    # Prior cleanup since spacy misses some stuff
    if remove_punct and remove_num:
        query = re.sub(r'[^a-zA-Z ]+', '', query)
    elif remove_punct:
        query = re.sub(r'[^a-zA-Z0-9 ]+', '', query)

    NLP = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    formatted_query = NLP(query)
    tokens = []
    for token in formatted_query:

        if remove_punct and token.is_punct:
            continue
        if remove_stop and token.is_stop:
            continue
        if remove_num and token.like_num:
            continue
        if allowed_pos and token.pos_ not in allowed_pos:
            continue

        if lemmatize:
            word = token.lemma_
        else:
            word = token.text
        
        if lowercase:
            word = word.lower()

        # Clean up if spacy missed them
        if remove_punct:
            word = re.sub(r'[^a-zA-Z]+', '', word)
        if remove_num:
            word = re.sub(r'[0-9]+', '', word)

        if word.strip():
            tokens.append(word)
    
    return ' '.join(tokens)

def term_query_vector(preprocessed_query : str, term_indexes : np.ndarray, boolean_vector : bool = False):
    """
    Returns the term vector representation of the prepocessed query. 
    
    Parameters:
    - preprocessed_query (str) : the query preprocessed as needed
    - term_indexes (np.ndarray) : the ordered set of all terms in the collection where we are retrieving
    - boolean_vector (bool) : if the output vector is a boolean one (presence or absence of the term in the query)
                              or a frequencies one (how many times the term is present in the query)
    """
    query_vector = np.zeros(term_indexes.shape)
    for term in preprocessed_query.split():
        idx = np.where(term_indexes == term)
        if len(idx) != 0: # if the term is present in the collection of document
            query_vector[idx[0]] += 1
            if boolean_vector:
                query_vector[idx[0]] = 1
    return query_vector

if __name__ == '__main__':
    indexes = np.array(['alpha', 'beta', 'gamma'])
    query = 'is alpha a beta delta?'
    p_q = preprocess_query_for_lsi(query)
    q_v = term_query_vector(p_q, indexes)
    print(q_v)
    
