import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Document:
    """Simple document class to store text content and metadata"""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(metadata={self.metadata})"

class TfidfVectorStore:
    """TF-IDF based vector store for document similarity search"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents = []
        self.tfidf_matrix = None

    def add_documents(self, documents):
        """Add documents to the vector store"""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self
    
    def save_local(self, folder_path):
        """Save the vector store to disk"""
        os.makedirs(folder_path, exist_ok=True)

        # Save the vectorizer
        with open(f"{folder_path}/vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        # Save the TFIDF matrix
        with open(f"{folder_path}/tfidf_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)

        # Save the documents
        with open(f"{folder_path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Vector store saved to {folder_path}")
    
    @classmethod
    def load_local(cls, folder_path):
        """Load vector store from disk"""
        vector_store = cls()

        # Load the vectorizer
        with open(f"{folder_path}/vectorizer.pkl", "rb") as f:
            vector_store.vectorizer = pickle.load(f)

        # Load the TFIDF matrix
        with open(f"{folder_path}/tfidf_matrix.pkl", "rb") as f:
            vector_store.tfidf_matrix = pickle.load(f)

        # Load the documents
        with open(f"{folder_path}/documents.pkl", "rb") as f:
            vector_store.documents = pickle.load(f)

        logger.info(f"Loaded vector store with {len(vector_store.documents)} documents")
        return vector_store
    
    def similarity_search(self, query, k=4):
        """Find similar documents to the query"""
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get indices of top k similar documents
        top_indices = similarities.argsort()[-k:][::-1]

        # Return the top k documents with scores
        results = []
        for i in top_indices:
            results.append({
                "document": self.documents[i],
                "score": similarities[i]
            })
        
        return results
    
    def get_relevant_documents(self, query, k=4):
        """Get relevant documents for a query (simplified retriever interface)"""
        results = self.similarity_search(query, k=k)
        return [result["document"] for result in results]