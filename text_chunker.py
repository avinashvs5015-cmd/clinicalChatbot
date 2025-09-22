"""
Text chunker module for RAG system.
Handles text splitting with overlap for better context preservation.
"""

import re
import logging
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Handle text chunking with configurable size and overlap."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str, source_document: str = "unknown") -> List[Dict]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            source_document: Source document path for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            logger.warning(f"Empty text provided for chunking from {source_document}")
            return []
        
        try:
            # Split the text
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only include non-empty chunks
                    chunk_obj = {
                        'text': chunk,
                        'metadata': {
                            'source_document': source_document,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk),
                            'start_char': text.find(chunk) if chunk in text else -1
                        }
                    }
                    chunk_objects.append(chunk_obj)
            
            logger.info(f"Created {len(chunk_objects)} chunks from text of length {len(text)}")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Error chunking text from {source_document}: {e}")
            return []
    
    def chunk_document(self, text: str, file_path: str) -> List[Dict]:
        """
        Chunk a single document.
        
        Args:
            text: Document text
            file_path: Path to the source file
            
        Returns:
            List of chunk dictionaries
        """
        return self.chunk_text(text, source_document=file_path)
    
    def chunk_documents(self, documents: Dict[str, str]) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: Dictionary mapping file paths to text content
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for file_path, text in documents.items():
            chunks = self.chunk_document(text, file_path)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} total chunks from {len(documents)} documents")
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'documents_processed': 0
            }
        
        chunk_sizes = [len(chunk['text']) for chunk in chunks]
        unique_documents = set(chunk['metadata']['source_document'] for chunk in chunks)
        
        stats = {
            'total_chunks': len(chunks),
            'total_characters': sum(chunk_sizes),
            'avg_chunk_size': sum(chunk_sizes) / len(chunks),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'documents_processed': len(unique_documents)
        }
        
        return stats
    
    def preview_chunks(self, chunks: List[Dict], max_chunks: int = 5) -> None:
        """
        Print a preview of the chunks for debugging.
        
        Args:
            chunks: List of chunk dictionaries
            max_chunks: Maximum number of chunks to preview
        """
        print(f"\nChunk Preview (showing up to {max_chunks} chunks):")
        print("=" * 60)
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk['metadata']['source_document']}")
            print(f"Size: {len(chunk['text'])} characters")
            print(f"Text preview: {chunk['text'][:200]}...")
            print("-" * 40)


# Example usage
if __name__ == "__main__":
    # Test the chunker
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    # Test text
    test_text = """
    This is a test document for chunking. It contains multiple paragraphs and sentences.
    
    The chunker should split this text into smaller pieces while maintaining some overlap
    between chunks to preserve context.
    
    Each chunk will have metadata about its source document and position within the
    original text.
    """
    
    chunks = chunker.chunk_text(test_text, "test_document.txt")
    chunker.preview_chunks(chunks)
    
    stats = chunker.get_chunk_statistics(chunks)
    print(f"\nChunk Statistics: {stats}")