"""
    This code is developed as final project for the Information Retrieval course (2024 - 2025).
    Authors : Bortolussi L., Bredariol F., Tonet L.
"""
from data_handler import *
from term_document_matrix import *
from lsi import *
from fftq_handler import *
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class LSI_IR:
    """
    The LSI_IR class is the wrapper for all that has been developed so far.
    We could take advantage of the possibility to load and store things around,
    but for now this class will always read the data, preprocess it, create
    the term document matrix, compute the svd and answer the query in this space.
    """
    def __init__(self, data_path : str, preprocess_protocol : tuple | None = None, metric : str = "freq", n_components : int = 100):
        """
            Parameters:
            - data_path (str) : where the collection is stored.
            - preprocess_protocol (tuple) : tuple of values used by the preprocessing for both the collection and queries.
              It is composed as follow: 
                    - text_columns (list) : The name of the columns containing the text to be preprocessed.
                    - lowercase (bool) : If True, convert text to lowercase.
                    - remove_punct (bool) : If True, remove punctuation tokens.
                    - remove_stop (bool) : If True, remove stopword tokens.
                    - lemmatize (bool) : If True, lemmatize tokens to their base form.
                    - remove_num (bool) : If True, remove number-like tokens.
                    - allowed_pos (list) : A list of Part-of-Speech tags to keep.
                                        Example: ['NOUN', 'PROPN', 'ADJ', 'VERB'].
                                        If None, all tokens are kept.
              IF SET TO NONE the default option will be considered.
            - metric (str) : what metric to use while computing the document matrix.
            - n_components (int) : number of components to consider for the svd.
        """
        self.preprocess_protocol = preprocess_protocol
        self.metric = metric
        self.n_components = n_components
        self.data_path = data_path

        self.parsed_df = parse_to_dataframe(self.data_path)
        self.document_indexes = np.array(self.parsed_df["T"])
        if preprocess_protocol == None:
            self.preprocessed_df = preprocess_for_lsi(self.parsed_df)
        else:
            self.preprocessed_df = preprocess_for_lsi(self.parsed_df, *self.preprocess_protocol)
        
        self.term_document_matrix, self.term_indexes = build_term_documents_mat(self.preprocessed_df, self.metric)

        self.latent_semantic_indexing = LSI(self.term_document_matrix, n_components=self.n_components, terms_indexes=self.term_indexes)

        self.term_tf_idf = compute_idf(self.term_document_matrix, compute_doc_freq(self.term_document_matrix))

    def retrieve(self, query : str, n_doc : int = 5):
        """
            Retrieve the n_doc most relevant documents from the collection linked to the query.
            
            Parameters:
            - query (str) : free form text query
            - n_doc (int) : number of documents to retrieve
        """

        n_doc = min(n_doc, len(self.document_indexes))

        if self.preprocess_protocol == None:
            preprocessed_query = preprocess_query_for_lsi(query)
        else:
            preprocess_protocol = list(self.preprocess_protocol)[1:] # we do not need the first part of this
            preprocessed_query = preprocess_query_for_lsi(query, *preprocess_protocol)

        query_vector = term_query_vector(preprocessed_query, self.term_indexes, "freq")

        if self.metric == "tf-idf":
            print(query_vector.shape)
            print(self.term_tf_idf.shape)
            query_vector = query_vector*self.term_tf_idf

        query_lsi = np.linalg.inv(np.diag(self.latent_semantic_indexing.concept_strength)) @ self.latent_semantic_indexing.term_concept_similarity.T @ (query_vector.reshape(-1, 1))
        
        similarities = cosine_similarity(query_lsi.T, self.latent_semantic_indexing.document_concept_similarity)

        doc_indexes = np.argsort(similarities[0])[::-1] # Sort by descending similarity

        for idx in doc_indexes[:n_doc]:
            print(f"Doc {idx} TITLE : {self.document_indexes[idx].upper()}\n \033[96m[Similarity: {similarities[0][idx]:.3f}]\033[97m")

        return doc_indexes[:n_doc]
    
    def save(self, path : str):
        """
            Method that saves the istance of the object.

            Parameters:
            -path : The file path as a string. It should not include a file extension (e.g., .txt, .csv), the function will handle adding the appropriate extension.
        """
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self, f)

    def interface(self):
        pass

def load(path) -> LSI_IR:
    """
            Method that loads one particular istance of an object of the clas LSI.

            Parameters:
            -path : The file path as a string. It should not include a file extension (e.g., .txt, .csv), the function will handle adding the appropriate extension.

            Ouputs:
            An object of the class LSI
        """
    with open(f"{path}.pkl", 'rb') as f:
        obj = pickle.load(f)
    return obj

if __name__ == '__main__':
    path = "source\\lsi_ir_saving_proof"
    if os.path.exists(f"{path}.pkl"):
        fr = load(path)
    else:
        fr = LSI_IR(data_path='data\\cran\\cran.all.1400', preprocess_protocol=None, metric="tf-idf", n_components=100)
        fr.save(path)
    fr.retrieve("buckling of circular cones under axial")