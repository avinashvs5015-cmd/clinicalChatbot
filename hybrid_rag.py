"""
Hybrid RAG System combining exact text search with vector embeddings
"""

import os
import re
from typing import List, Dict, Any, Optional
from document_extractor import DocumentExtractor
from text_chunker import TextChunker
from vector_store import VectorStore


class HybridRAG:
    def __init__(self, documents_dir: str = "./documents"):
        """
        Initialize Hybrid RAG system with persistent embeddings
        
        Args:
            documents_dir: Directory containing documents to process
        """
        self.documents_dir = documents_dir
        
        # Initialize components
        self.document_extractor = DocumentExtractor()
        self.text_chunker = TextChunker()
        
        # Initialize vector store with persistent embeddings
        self.vector_store = VectorStore()
        
        # Initialize text-based chunks for exact search
        self.text_chunks = []
        
        # Load and process documents using persistent cache
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the RAG system with persistent embeddings"""
        print("Initializing Hybrid RAG system...")
        
        # Update embeddings using persistent cache (only processes new/modified files)
        embedding_stats = self.vector_store.update_embeddings(self.documents_dir)
        
        # Load text chunks for exact search
        self.load_text_chunks()
        
        print(f"Hybrid RAG system initialized successfully")
        print(f"Vector embeddings: {embedding_stats['total_embeddings_in_db']}")
        print(f"Text chunks for exact search: {len(self.text_chunks)}")
    
    def load_text_chunks(self):
        """Load text chunks for exact text search"""
        self.text_chunks = []
        
        # Get all document files
        for root, dirs, files in os.walk(self.documents_dir):
            for file in files:
                if file.lower().endswith(('.txt', '.pdf', '.docx', '.md')):
                    file_path = os.path.join(root, file)
                    
                    # Extract text
                    text_content = self.document_extractor.extract_text(file_path)
                    if text_content.strip():
                        # Chunk the text
                        chunks = self.text_chunker.chunk_text(text_content, file_path)
                        self.text_chunks.extend(chunks)
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query for enhanced search"""
        # Remove common words and extract meaningful terms
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'under', 'between', 'among',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'what', 'who', 'where', 'when', 'why', 'how', 'which', 'that',
            'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Split and clean terms
        terms = re.findall(r'\b\w+\b', query.lower())
        key_terms = [term for term in terms if term not in common_words and len(term) > 2]
        
        # Also include quoted phrases and names (capital letters)
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        key_terms.extend([phrase.lower() for phrase in quoted_phrases])
        key_terms.extend([name.lower() for name in names])
        
        return list(set(key_terms))  # Remove duplicates
    
    def enhanced_exact_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced exact text search with key term extraction"""
        query_lower = query.lower()
        key_terms = self.extract_key_terms(query)
        
        scored_chunks = []
        
        for chunk in self.text_chunks:
            text_lower = chunk['text'].lower()
            score = 0
            
            # Exact phrase match (highest weight)
            if query_lower in text_lower:
                score += 10
            
            # Key term matches
            for term in key_terms:
                if term in text_lower:
                    # Count occurrences
                    count = text_lower.count(term)
                    score += count * 2
            
            # Word boundary matches (more precise)
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 2:
                    pattern = r'\b' + re.escape(word) + r'\b'
                    matches = len(re.findall(pattern, text_lower))
                    score += matches * 1.5
            
            if score > 0:
                scored_chunks.append({
                    'text': chunk['text'],
                    'metadata': chunk.get('metadata', {}),
                    'exact_score': score
                })
        
        # Sort by score and return top results
        scored_chunks.sort(key=lambda x: x['exact_score'], reverse=True)
        return scored_chunks[:top_k]
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector-based semantic search"""
        return self.vector_store.search(query, k=top_k)
    
    def hybrid_search(self, query: str, exact_weight: float = 0.6, vector_weight: float = 0.4, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining exact text matching with vector similarity
        
        Args:
            query: Search query
            exact_weight: Weight for exact search results
            vector_weight: Weight for vector search results
            top_k: Number of results to return
            
        Returns:
            List of ranked search results
        """
        # Get results from both search methods
        exact_results = self.enhanced_exact_search(query, top_k * 2)
        vector_results = self.vector_search(query, top_k * 2)
        
        # Combine and score results
        combined_results = {}
        
        # Add exact search results
        for i, result in enumerate(exact_results):
            text = result['text']
            if text not in combined_results:
                combined_results[text] = {
                    'text': text,
                    'metadata': result['metadata'],
                    'exact_score': result['exact_score'],
                    'vector_score': 0.0,
                    'exact_rank': i + 1
                }
        
        # Add vector search results
        for i, result in enumerate(vector_results):
            text = result['text']
            if text in combined_results:
                combined_results[text]['vector_score'] = result.get('similarity', 0.0)
                combined_results[text]['vector_rank'] = i + 1
            else:
                combined_results[text] = {
                    'text': text,
                    'metadata': result['metadata'],
                    'exact_score': 0.0,
                    'vector_score': result.get('similarity', 0.0),
                    'vector_rank': i + 1
                }
        
        # Calculate combined scores
        final_results = []
        for result in combined_results.values():
            # Normalize scores
            exact_norm = result['exact_score'] / max(1, max([r['exact_score'] for r in combined_results.values()]))
            vector_norm = result['vector_score']
            
            # Calculate combined score
            combined_score = (exact_weight * exact_norm) + (vector_weight * vector_norm)
            
            result['combined_score'] = combined_score
            final_results.append(result)
        
        # Sort by combined score and return top results
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return final_results[:top_k]
    
    def search(self, query: str, search_type: str = "hybrid", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents using the specified method
        
        Args:
            query: Search query
            search_type: Type of search ('exact', 'vector', 'hybrid')
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if search_type == "exact":
            return self.enhanced_exact_search(query, top_k)
        elif search_type == "vector":
            return self.vector_search(query, top_k)
        elif search_type == "hybrid":
            return self.hybrid_search(query, top_k=top_k)
        else:
            raise ValueError(f"Invalid search_type: {search_type}. Use 'exact', 'vector', or 'hybrid'")
    
    def get_context(self, query: str, search_type: str = "hybrid", top_k: int = 3) -> str:
        """
        Get relevant context for the query
        
        Args:
            query: Search query
            search_type: Type of search to use
            top_k: Number of chunks to include in context
            
        Returns:
            Formatted context string
        """
        results = self.search(query, search_type, top_k)
        
        if not results:
            return "No relevant information found in the documents."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Context {i}:\n{result['text']}")
        
        return "\n\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    # Initialize hybrid RAG system
    rag = HybridRAG()
    
    # Example searches
    queries = [
        "machine learning",
        "MR.DINESH RAGHUWANSI",
        "What is artificial intelligence?"
    ]
    
    for query in queries:
        print(f"\n--- Query: {query} ---")
        
        # Try different search types
        for search_type in ["exact", "vector", "hybrid"]:
            print(f"\n{search_type.upper()} Search:")
            results = rag.search(query, search_type, top_k=2)
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.get('exact_score', 0):.2f} | "
                      f"Vector: {result.get('vector_score', 0):.2f} | "
                      f"Combined: {result.get('combined_score', 0):.2f}")
                print(f"     Text: {result['text'][:100]}...")