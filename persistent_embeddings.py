"""
Persistent Embedding Cache System
Stores embeddings with file metadata to avoid re-processing unchanged files
"""

import os
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from document_extractor import DocumentExtractor
from text_chunker import TextChunker


class PersistentEmbeddingCache:
    """
    Manages persistent storage of document embeddings with intelligent caching
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 metadata_file: str = "./embedding_metadata.json",
                 collection_name: str = "document_embeddings"):
        """
        Initialize the persistent embedding cache
        
        Args:
            persist_directory: Directory to store ChromaDB data
            metadata_file: File to store file metadata for change detection
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.metadata_file = metadata_file
        self.collection_name = collection_name
        
        # Create directories if they don't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize document processing components
        self.document_extractor = DocumentExtractor()
        self.text_chunker = TextChunker()
        
        # Load or create metadata tracking
        self.file_metadata = self.load_metadata()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection '{collection_name}' with {self.collection.count()} embeddings")
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection '{collection_name}'")
    
    def load_metadata(self) -> Dict:
        """Load file metadata from disk"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata file: {e}")
                return {}
        return {}
    
    def save_metadata(self):
        """Save file metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.file_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata file: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def get_file_metadata(self, file_path: str) -> Dict:
        """Get current file metadata (size, modification time, hash)"""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'hash': self.get_file_hash(file_path)
            }
        except Exception:
            return {}
    
    def has_file_changed(self, file_path: str) -> bool:
        """Check if file has changed since last processing"""
        current_metadata = self.get_file_metadata(file_path)
        stored_metadata = self.file_metadata.get(file_path, {})
        
        # Compare hash (most reliable) and modification time
        return (current_metadata.get('hash') != stored_metadata.get('hash') or
                current_metadata.get('mtime') != stored_metadata.get('mtime'))
    
    def remove_file_embeddings(self, file_path: str):
        """Remove embeddings for a specific file"""
        try:
            # Get all embeddings for this file
            results = self.collection.get(
                where={"file_path": file_path}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"Removed {len(results['ids'])} embeddings for {file_path}")
        except Exception as e:
            print(f"Warning: Could not remove embeddings for {file_path}: {e}")
    
    def process_file(self, file_path: str) -> int:
        """Process a single file and store its embeddings"""
        print(f"Processing: {file_path}")
        
        # Extract text from document
        text_content = self.document_extractor.extract_text(file_path)
        if not text_content.strip():
            print(f"Warning: No text content extracted from {file_path}")
            return 0
        
        # Chunk the text
        chunks = self.text_chunker.chunk_text(text_content, file_path)
        if not chunks:
            print(f"Warning: No chunks created from {file_path}")
            return 0
        
        # Remove old embeddings for this file
        self.remove_file_embeddings(file_path)
        
        # Create embeddings for chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts).tolist()
        
        # Prepare data for ChromaDB
        ids = [f"{file_path}_{i}" for i in range(len(chunks))]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                'file_path': file_path,
                'chunk_index': i,
                'text_length': len(chunk['text'])
            }
            # Add any additional metadata from the chunk
            if 'metadata' in chunk:
                metadata.update(chunk['metadata'])
            metadatas.append(metadata)
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update file metadata
        self.file_metadata[file_path] = self.get_file_metadata(file_path)
        
        print(f"Stored {len(chunks)} embeddings for {file_path}")
        return len(chunks)
    
    def update_embeddings(self, documents_dir: str = "./documents") -> Dict:
        """
        Update embeddings for all documents in the directory
        Only processes new or modified files
        """
        print("Checking for new or modified documents...")
        
        # Get all document files
        document_files = []
        supported_extensions = {'.txt', '.pdf', '.docx', '.md'}
        
        for root, dirs, files in os.walk(documents_dir):
            for file in files:
                if Path(file).suffix.lower() in supported_extensions:
                    file_path = os.path.join(root, file)
                    document_files.append(file_path)
        
        # Check which files need processing
        files_to_process = []
        for file_path in document_files:
            if self.has_file_changed(file_path):
                files_to_process.append(file_path)
        
        # Remove embeddings for files that no longer exist
        existing_files = set(document_files)
        stored_files = set(self.file_metadata.keys())
        deleted_files = stored_files - existing_files
        
        for deleted_file in deleted_files:
            print(f"Removing embeddings for deleted file: {deleted_file}")
            self.remove_file_embeddings(deleted_file)
            del self.file_metadata[deleted_file]
        
        # Process new/modified files
        stats = {
            'total_files': len(document_files),
            'processed_files': len(files_to_process),
            'skipped_files': len(document_files) - len(files_to_process),
            'deleted_files': len(deleted_files),
            'total_embeddings': 0
        }
        
        if files_to_process:
            print(f"Processing {len(files_to_process)} new/modified files...")
            for file_path in files_to_process:
                chunks_added = self.process_file(file_path)
                stats['total_embeddings'] += chunks_added
        else:
            print("No new or modified files found. Using cached embeddings.")
        
        if deleted_files:
            print(f"Removed embeddings for {len(deleted_files)} deleted files")
        
        # Save updated metadata
        self.save_metadata()
        
        # Get final collection count
        total_embeddings = self.collection.count()
        stats['total_embeddings_in_db'] = total_embeddings
        
        print(f"\nEmbedding update complete:")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Processed: {stats['processed_files']}")
        print(f"  Skipped (cached): {stats['skipped_files']}")
        print(f"  Deleted: {stats['deleted_files']}")
        print(f"  Total embeddings in database: {stats['total_embeddings_in_db']}")
        
        return stats
    
    def get_collection(self):
        """Get the ChromaDB collection for use in other components"""
        return self.collection
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of matching documents with metadata
        """
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def reset_cache(self):
        """Reset the entire embedding cache (useful for debugging)"""
        print("Resetting embedding cache...")
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            self.file_metadata = {}
            self.save_metadata()
            print("Embedding cache reset successfully")
        except Exception as e:
            print(f"Error resetting cache: {e}")


def main():
    """Example usage and testing"""
    cache = PersistentEmbeddingCache()
    
    # Update embeddings (only processes new/modified files)
    stats = cache.update_embeddings()
    
    # Example search
    if stats['total_embeddings_in_db'] > 0:
        print("\nTesting search functionality:")
        results = cache.search("machine learning", n_results=3)
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['text'][:100]}...")


if __name__ == "__main__":
    main()