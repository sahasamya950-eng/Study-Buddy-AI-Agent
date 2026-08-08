"""
Vector Store & RAG Retrieval module using FAISS and embeddings.
Provides semantic chunking and relevant passage search for large study documents.
"""

import re
from typing import List, Dict, Any, Tuple
from utils.logger import logger

class StudyVectorStore:
    """FAISS and TF-IDF fallback vector store for document indexing and semantic retrieval."""
    
    def __init__(self):
        self.chunks: List[str] = []
        self.faiss_index = None
        self.encoder = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.is_indexed = False

    def build_index(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> bool:
        """Splits document text into overlapping chunks and indexes them."""
        if not text or len(text.strip()) < 20:
            return False

        # Clean and split into overlapping chunks
        words = text.split()
        self.chunks = []
        
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                self.chunks.append(chunk.strip())
            i += (chunk_size - chunk_overlap)

        if not self.chunks:
            self.chunks = [text[:1000]]

        # Try FAISS with Sentence Transformers
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np

            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self.encoder.encode(self.chunks, convert_to_numpy=True)
            
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings.astype(np.float32))
            self.is_indexed = True
            logger.info(f"Successfully indexed {len(self.chunks)} chunks into FAISS vector store.")
            return True
        except Exception as e:
            logger.info(f"SentenceTransformers/FAISS not available ({e}). Building Scikit-Learn TF-IDF vector index...")

        # Fallback to TF-IDF vectorizer
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            self.is_indexed = True
            logger.info(f"Indexed {len(self.chunks)} chunks into TF-IDF vector store.")
            return True
        except Exception as err2:
            logger.warning(f"TF-IDF indexing fallback failed: {err2}. Using simple substring search.")
            self.is_indexed = True
            return True

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Retrieves top_k relevant text chunks matching user query."""
        if not self.chunks:
            return ""

        top_k = min(top_k, len(self.chunks))

        # FAISS search
        if self.faiss_index and self.encoder:
            try:
                import numpy as np
                query_vec = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
                distances, indices = self.faiss_index.search(query_vec, top_k)
                retrieved = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
                return "\n\n---\n\n".join(retrieved)
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}")

        # TF-IDF search
        if self.vectorizer and self.tfidf_matrix is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                query_vec = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
                top_indices = similarities.argsort()[-top_k:][::-1]
                retrieved = [self.chunks[idx] for idx in top_indices]
                return "\n\n---\n\n".join(retrieved)
            except Exception as e:
                logger.warning(f"TF-IDF search failed: {e}")

        # Keyword match fallback
        query_words = set(re.findall(r'\w+', query.lower()))
        scored_chunks = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\w+', chunk.lower()))
            overlap = len(query_words.intersection(chunk_words))
            scored_chunks.append((overlap, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [item[1] for item in scored_chunks[:top_k]]
        return "\n\n---\n\n".join(top_chunks)

# Global singleton instance
vector_store = StudyVectorStore()
