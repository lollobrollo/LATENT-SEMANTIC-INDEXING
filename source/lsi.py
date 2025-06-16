"""
    This code is developed as final project for the Information Retrieval course (2024 - 2025).
    Authors : Bortolussi L., Bredariol F., Tonet L.
"""

from svd import *
import numpy as np
import pickle
from matplotlib import pyplot as plt

class LSI:
    """
        This class stores all methods that have to do with LSI given a document term matrix.
        Since there is an external method that loads a precomputed object of this class, it can be used instead of the initializer to avoid recomputing the SVD.
    """

    def __init__(self, term_document_matrix : np.ndarray, n_components : int = 100, document_indexes : np.ndarray | None = None, terms_indexes : np.ndarray | None = None):
        """
            Parameters:
            - document_term_matrix : Document term matrix, assumed to be an np array
            - n_components : number of components of the SVD to be stored 
            - document_indexes : array linking the index of a document in the document term matrix to its title
           - term_indexes : array linking the index of a term in the document term matrix to its value

            Note : if document indexes and term indexes are none, document and terms will be just referred to as numbers (a default array will be created). 
            Otherwise their titles/values will be visualized.
        """
        self.term_document_matrix = term_document_matrix
        self.n_components = n_components
        self.document_indexes = document_indexes if document_indexes is not None else np.array([i for i in range (term_document_matrix.shape[0])])
        self.term_indexes = terms_indexes if terms_indexes is not None else  np.array([i for i in range (term_document_matrix.shape[1])])
        self.compute_lsi()
    
    def compute_lsi(self):
        """
            Perform the truncated svd on the term document matrix in order to obtain:
            1. the document concept similarity matrix (lsi matrix)
            2. the concept strength vector (singular values)
            3. the term concept similarity matrix (principal components)
        """
        # perform svd and returns U*S object. From this object one can obtain S = diag(obj.singular_values_) and V* = obj.components_
        # so U will be U * S / obj.singular_values_
        self.term_concept_similarity, self.s_v_d = perform_svd(self.term_document_matrix, self.n_components)
        self.term_concept_similarity = self.term_concept_similarity / self.s_v_d.singular_values_
        self.concept_strength = self.s_v_d.singular_values_
        self.document_concept_similarity = self.s_v_d.components_.T
    
    def get_lsi_matrices(self):
        """
            Returns the lsi matrices, the singular values and the full object that is the truncated svd.
            Note that the core of lsi is the document concept similarity, since it stores the representation of each document
            in the new projected space, in which we will compute the cosine similarity.

            In order, this returns:
            1. the document concept similarity matrix (lsi matrix) [numpy array]
            2. the term concept similarity matrix (principal components) [numpy array]
            3. the concept strength vector (singular values) [numpy array]
            4. the truncated svd object [TRUNCATED SVD object]
        """
        return self.document_concept_similarity, self.term_concept_similarity, self.concept_strength, self.s_v_d
    
    def analyze_lsi_matrices(self, concept_index_1 : int = 0, concept_index_2 : int = 1):
        """
        This function analyzes the LSI matrix by showing how documents are projected onto the reduced space.
        Specifically, it visualizes the projection of all documents in a 2D space defined by two requested concept indices.
        The concept indices must be valid—that is, they cannot exceed the total number of components used in the SVD computation.
        """

        if concept_index_1 >= self.n_components or concept_index_2 >= self.n_components:
            print(f"\033[93mError. Concept index too large! It must be in the range [0, {self.n_components - 1}]\033[97m ")
            raise ValueError
        
        x = self.document_concept_similarity[:, concept_index_1] # Values for Concept 1
        y = self.document_concept_similarity[:, concept_index_2] # Values for Concept 2

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='orange', label='Documents')

        for i, (x_val, y_val) in enumerate(zip(x, y)):
            plt.text(x_val + 0.02, y_val, f'Doc {self.document_indexes[i]}', fontsize=9)

        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.title(f"Document-Concept Similarity in the Projected space defined by the concepts {concept_index_1}, {concept_index_2}")
        plt.xlabel(f"Concept {concept_index_1}")
        plt.ylabel(f"Concept {concept_index_2}")
        plt.grid()
        plt.legend()
        plt.show()

    def analyze_lsi_concepts_composition(self, concept_index : int = 0, n_terms : int = 20):
        """
            This function analyzes how concepts are defined wrt the terms, by showing term weights for a given concept.
            Concept index must be valid, so it can not exceed the total number of components used to compute the svd.

            Parameters:
            - concept_index : the index of the concept to analyze
            - n_terms : the number of terms to take in consideration (the n with the greatest weights magnitude)
        """
        if concept_index >= self.n_components:
            print(f"\033[93mError. Concept index too large! It must be in the range [0, {self.n_components - 1}]\033[97m ")
            raise ValueError
        
        concept_weights = abs(self.term_concept_similarity.T[concept_index])
        sorted_indexing = np.argsort(concept_weights)
        interested_indexes = sorted_indexing[-n_terms:]

        plt.figure(figsize=(8, 6))
        
        plt.barh([i for i in range (n_terms, 0, -1)], self.term_concept_similarity.T[concept_index][interested_indexes], tick_label = self.term_indexes[interested_indexes], color = 'orange')
        #plt.barh(self.term_indexes[interested_indexes], concept_weights[interested_indexes], color = 'orange')
        plt.title(f"Term Weights for Concept {concept_index}")
        plt.xlabel(f"Weights")
        plt.show()
    
    def save(self, path : str):
        """
            Method to save an istance of this object.

            Parameters:
            - path : The file path as a string. It should not include a file extension (e.g., .txt, .csv), the function will handle adding the appropriate extension.
        """
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self, f)

    def __str__(self):
        """
            Return a simple string representation of the object.
        """
        s = f"DOCUMENT CONCEPT SIMILARITY MATRIX\n{self.document_concept_similarity}\nTERM CONCEPT SIMILARITY MATRIX\n{self.term_concept_similarity}\nCONCEPT STRENGTH VECTOR\n{self.concept_strength}"
        return s
    
def load(path : str) -> LSI:
    """
        Methods to load one particular istance of an object of the clas LSI.

        Parameters:
        -path is the file path. It must be a str and it should end without any extension (eg .txt, .csv). The extension will be handled by the function.

        Ouputs:
        An object of the class LSI
    """
    with open(f"{path}.pkl", 'rb') as f:
        obj = pickle.load(f)
    return obj

if __name__ == '__main__':

    """tdm = np.array([[0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1]])
    document_indexes = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    term_indexes = np.array(['alpha', 'beta', 'casa', 'delta'])
    lsi_handler = LSI(tdm, n_components=2, document_indexes=document_indexes, terms_indexes=term_indexes)
    lsi_handler.analyze_lsi_matrices()
    lsi_handler.analyze_lsi_concepts_composition(concept_index=0)
    lsi_handler.save("./source/lsi_saving_proof")
    lsi_handler = load("./source/lsi_saving_proof")
    print("---")
    print(lsi_handler)
    lsi_handler = load("./source/lsi_saving_proof")
    lsi_handler.analyze_lsi_matrices()
    lsi_handler.analyze_lsi_concepts_composition()"""
    from term_document_matrix import *
    from data_handler import *
    df = parse_to_dataframe('data\\cat encyclopedia\\documents.all.5')
    df = preprocess_for_lsi(df)
    tdm, term_indexes = build_term_documents_mat(df)
    lsi_handler = LSI(tdm, n_components = 5, terms_indexes=term_indexes)
    lsi_handler.analyze_lsi_concepts_composition(0)
    lsi_handler.analyze_lsi_concepts_composition(1)
    #lsi_handler.analyze_lsi_matrices(concept_index_1=0, concept_index_2=1)
    