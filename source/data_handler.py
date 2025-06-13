"""
    This code is developed as final project for the Information Retrieval course (2024 - 2025).
    Authors : Bortolussi L., Bredariol F., Tonet L.
"""


import pandas as pd
import spacy
from tqdm import tqdm


# File format considdered: dot-tagged dataset
# Each document is encoded as follows:
# .I <id>  → Document ID
# .T       → Title (1+ lines)
# .A       → Author(s)
# .B       → Bibliographic info
# .W       → Abstract / full text (1+ lines)


def parse_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Parses a dot-tagged document file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the input file (Cranfield-style format).

    Returns:
    - pd.DataFrame: DataFrame with columns ['I', 'T', 'A', 'B', 'W'] for ID, Title, Author, Bibliographic info, and Abstract.
    """

    with open(file_path, 'r') as f:
        content = f.read()

    documents = []
    current_doc = {}
    lines = content.split('\n')
    current_field = None

    for line in lines:
        if line.startswith('.I'):
            if current_doc:
                documents.append(current_doc)
            current_doc = {'I': line[3:].strip()}
            current_field = None
        elif line.startswith('.T'):
            current_field = 'T'
            current_doc[current_field] = ''
        elif line.startswith('.A'):
            current_field = 'A'
            current_doc[current_field] = ''
        elif line.startswith('.B'):
            current_field = 'B'
            current_doc[current_field] = ''
        elif line.startswith('.W'):
            current_field = 'W'
            current_doc[current_field] = ''
        elif current_field:
            current_doc[current_field] += line + ' '

    # Save also the last document
    if current_doc:
        documents.append(current_doc)

    # Strip whitespace from all fields
    for doc in documents:
        for key in doc:
            doc[key] = doc[key].strip()

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(documents)
    return df



def preprocess_for_lsi(
    df: pd.DataFrame,
    text_columns: list = ['W'],
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stop: bool = True,
    lemmatize: bool = True,
    remove_num: bool = True,
    allowed_pos: list = None
) -> pd.DataFrame:
    """
    Performs  preprocessing for our LSI model.

    Parameters:
    - df (pd.DataFrame) : Input DataFrame.
    - text_columns (list) : The name of the columns containing the text.
    - lowercase (bool) : If True, convert text to lowercase.
    - remove_punct (bool) : If True, remove punctuation tokens.
    - remove_stop (bool) : If True, remove stopword tokens.
    - lemmatize (bool) : If True, lemmatize tokens to their base form.
    - remove_num (bool) : If True, remove number-like tokens.
    - allowed_pos (list) : A list of Part-of-Speech tags to keep.
                          Example: ['NOUN', 'PROPN', 'ADJ', 'VERB'].
                          If None, all tokens are kept.

    Returns:
    - pd.DataFrame : DataFrame with a new 'clean_text' column.
    """
    if allowed_pos is None:
        allowed_pos = ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'] # Default meaningful POS

    df_copy = df.copy()

    # Pipeline that will do the hard work for us
    NLP = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    # List used to store data from all columns required
    all_texts = {key: [] for key in text_columns}

    for text_column in text_columns:
        print(f'Now processing column {text_column} ...')
        # Process texts in batches using nlp.pipe() for performance
        # list() turns the tqdm iterator (which wraps the nlp.pipe generator) into a list
        docs = list(tqdm(NLP.pipe(df_copy[text_column].fillna("").astype(str)), total=len(df_copy)))
        
        for doc in docs:
            tokens = []
            for token in doc:
                # Check conditions for skipping a token
                if remove_punct and token.is_punct:
                    continue
                if remove_stop and token.is_stop:
                    continue
                if remove_num and token.like_num:
                    continue
                if allowed_pos and token.pos_ not in allowed_pos:
                    continue

                # Determine the final form of the word
                if lemmatize:
                    word = token.lemma_
                else:
                    word = token.text
                
                if lowercase:
                    word = word.lower()

                # Safeguard that checks if there's any actual content left in the word
                if word.strip():
                    tokens.append(word)
            
            all_texts[text_column].append(" ".join(tokens))

    # Turn the dictionary in a list, put together text of the same docs with zip() and then join them into the new column
    df_copy['clean_text'] = [ ' '.join(elems) for elems in zip(*[all_texts[key] for key in all_texts])]
    return df_copy


# Here are auxiliary functions, used during developement, to save the docs dataframe to a file and later read it

def write_to_parquet(df: pd.DataFrame, file_path: str):
    """
    Writes a pandas DataFrame to a Parquet file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to write.
    - file_path (str): Path to the output .parquet file.
    """
    df.to_parquet(file_path, index=False)


def read_from_parquet(file_path: str) -> pd.DataFrame:
    """
    Reads a Parquet file into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the .parquet file.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_parquet(file_path)


# # # # # # = = = = = = = = = = = = = = - - - - - - - - - - - - - - - = = = = = = = = = = = = = = # # # # # #

# Example usage
if __name__ == '__main__':
    print("Parsing the docs...")
    df = parse_to_dataframe('LSI\\data\\cran\\cran.all.1400')
    print("Parsing complete.\nMoving on to preprpocessing...")
    # print(df.head())
    df = preprocess_for_lsi(df, text_columns=['T', 'W'])
    print("Preprocessing complete.\n")
    print(df.loc[5, 'W'],'\n')
    print(df.loc[5,'clean_text'])

    # Test if saving to file works as expected
    print("\nNow writing to file...")
    write_to_parquet(df, '.\\data\\test\\test.parquet')
    print("Now reading from file...\n")
    df = read_from_parquet('.\\data\\test\\test.parquet')
    print(df.loc[5,'clean_text'])