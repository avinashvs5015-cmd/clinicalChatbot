"""
Vector Store for Document RAG
Handles embedding storage and retrieval using persistent ChromaDB
"""

from typing import List, Dict, Any
from persistent_embeddings import PersistentEmbeddingCache

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB vector store with persistence"""
        self.persist_directory = persist_directory
        
        # Use the persistent embedding cache system
        self.embedding_cache = PersistentEmbeddingCache(
            persist_directory=persist_directory,
            collection_name="document_embeddings"
        )
        
        # Get the collection from the cache
        self.collection = self.embedding_cache.get_collection()
        
        # Initialize embedding model (for compatibility)
        self.embedding_model = self.embedding_cache.embedding_model
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store using persistent cache"""
        # This method is now handled by the persistent embedding cache
        # during the update_embeddings process
        print("Note: Document addition is now handled by PersistentEmbeddingCache.update_embeddings()")
        return len(chunks)
    
    def update_embeddings(self, documents_dir: str = "./documents"):
        """Update embeddings using the persistent cache system"""
        return self.embedding_cache.update_embeddings(documents_dir)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using the persistent cache"""
        results = self.embedding_cache.search(query, n_results=k)
        
        # Convert to expected format (distance -> similarity)
        for result in results:
            if 'distance' in result:
                result['similarity'] = 1 - result['distance']
            else:
                result['similarity'] = 0.0
        
        return results
    
    def get_collection_count(self) -> int:
        """Get the total number of embeddings in the collection"""
        return self.collection.count()
    
    def reset_cache(self):
        """Reset the entire embedding cache"""
        self.embedding_cache.reset_cache()