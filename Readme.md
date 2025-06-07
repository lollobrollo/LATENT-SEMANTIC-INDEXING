# LATENT SEMANTIC INDEXING
---

This is the final project for the 2024/2025 course in "information retrieval".

**Latent Semantic Indexing (LSI)** is a technique used in information retrieval and natural language processing to discover hidden (latent) relationships between words and documents. It aims to overcome the limitations of traditional keyword-based search by capturing the semantic meaning behind terms.

At its core, LSI works by:

1. Creating a term-document matrix, where each row represents a unique term, each column represents a document, and each cell contains the frequency (or TF-IDF score) of the term in the document.
2. Applying Singular Value Decomposition (SVD) to this matrix. SVD reduces the high-dimensional term space into a lower-dimensional semantic space, where similar words and documents are grouped closer together based on their usage patterns—not just their exact terms.
3. Mapping both documents and queries into this semantic space, allowing the system to find relevant results even when the query and the documents don’t share exact keywords.

Our work is divided in various part. In particular we defined 7 files:

1. **data_handler.py** which stores methods to import, handle and preprocess data
2. **term_document_matrix.py** which stores methods to create the term document matrix
3. **svd.py** which stores methods to perform the singol value decomposition and dimensionality reduction
4. **lsi.py** which stores methods to analyze, use and evaluate the latent semantic indexing
5. **fftq_handler.py** which stores methods to handle free form text queries
6. **lsi_ir.py** which stores methods to define an information retrieval system that uses lsi
7. **project.ipynb** which serves as final wrapper for everything

*Authors : Bortolussi L., Bredariol F., Tonet L.*
