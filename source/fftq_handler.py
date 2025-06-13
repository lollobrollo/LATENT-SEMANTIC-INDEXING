"""
    This code is developed as final project for the Information Retrieval course (2024 - 2025).
    Authors : Bortolussi L., Bredariol F., Tonet L.
"""

import spacy
import tqdm

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

        if word.strip():
            tokens.append(word)
    
    return ' '.join(tokens)

def term_query_vector(preprocessed_query : str, list_of_terms : list, mode : str = 'one_hot'):
    """
    Returns the term vector representation of the prepocessed query. 
    It is relevant how the term vector representation is computed. 
    """
    pass

if __name__ == '__main__':
    query = "Does everyone know something except me?"
    print(preprocess_query_for_lsi(query))
    
