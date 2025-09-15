import os
import re
import nltk
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
import warnings

# Suppress warnings that might cause issues
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# PDF processing
import pymupdf4llm

# Embeddings - handle gracefully
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning(f"SentenceTransformers not available: {e}")

# Vector database
import chromadb
from chromadb.config import Settings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for chunking strategy"""
    strategy: str = "sentence"  # "sentence", "fixed_size", "paragraph", "sliding_window"
    max_chunk_size: int = 500  # Maximum characters per chunk
    overlap_size: int = 50     # Overlap for sliding window
    min_chunk_size: int = 50   # Minimum characters per chunk

class PDFVectorPipeline:
    """Complete pipeline for PDF to vector database"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_path: str = "./vector_db",
                 collection_name: str = "pdf_chunks"):
        
        self.embedding_model_name = embedding_model
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                self.embeddings_available = True
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self.embedding_model = None
                self.embeddings_available = False
        else:
            logger.warning("SentenceTransformers not available, vector search will be disabled")
            self.embedding_model = None
            self.embeddings_available = False
        
        # Initialize vector database
        self.setup_vector_db()
        
    def setup_vector_db(self):
        """Initialize ChromaDB client and collection"""
        logger.info("Setting up ChromaDB...")
        
        # Create ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.vector_db_path)
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {self.collection_name}")
        except Exception:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pymupdf4llm"""
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Extract text using pymupdf4llm
            text = pymupdf4llm.to_markdown(pdf_path)
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        logger.info("Preprocessing text...")
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        
        # Clean up markdown formatting if needed
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove markdown headers
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def chunk_by_sentences(self, text: str, config: ChunkConfig) -> List[str]:
        """Chunk text by sentences"""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max size, save current chunk
            if len(current_chunk) + len(sentence) > config.max_chunk_size and current_chunk:
                if len(current_chunk.strip()) >= config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_fixed_size(self, text: str, config: ChunkConfig) -> List[str]:
        """Chunk text by fixed character size"""
        chunks = []
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 > config.max_chunk_size and current_chunk:
                if len(current_chunk.strip()) >= config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk += " " + word if current_chunk else word
        
        if current_chunk.strip() and len(current_chunk.strip()) >= config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, config: ChunkConfig) -> List[str]:
        """Chunk text by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) > config.max_chunk_size and current_chunk:
                if len(current_chunk.strip()) >= config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk.strip() and len(current_chunk.strip()) >= config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_text(self, text: str, config: ChunkConfig) -> List[str]:
        """Chunk text based on the specified strategy"""
        logger.info(f"Chunking text using strategy: {config.strategy}")
        
        if config.strategy == "sentence":
            chunks = self.chunk_by_sentences(text, config)
        elif config.strategy == "fixed_size":
            chunks = self.chunk_by_fixed_size(text, config)
        elif config.strategy == "paragraph":
            chunks = self.chunk_by_paragraphs(text, config)
        else:
            raise ValueError(f"Unknown chunking strategy: {config.strategy}")
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
            all_embeddings.extend(embeddings)
        
        return np.array(all_embeddings)
    
    def store_in_vector_db(self, chunks: List[str], embeddings: np.ndarray, 
                          metadata: List[Dict] = None) -> None:
        """Store chunks and embeddings in ChromaDB"""
        logger.info("Storing chunks and embeddings in vector database...")
        
        if metadata is None:
            metadata = [{"chunk_id": i, "length": len(chunk)} for i, chunk in enumerate(chunks)]
        
        # Generate unique IDs for each chunk
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Store in ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Successfully stored {len(chunks)} chunks in vector database")
    
    def search_similar(self, query: str, top_k: int = 5) -> Dict:
        """Search for similar chunks using vector similarity"""
        logger.info(f"Searching for similar chunks to query: '{query[:50]}...'")
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return {
            "query": query,
            "results": [
                {
                    "chunk": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        }
    
    def process_pdf(self, pdf_path: str, chunk_config: ChunkConfig = None) -> Dict:
        """Complete pipeline to process PDF and store in vector database"""
        if chunk_config is None:
            chunk_config = ChunkConfig()
        
        try:
            # Step 1: Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Step 2: Preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Step 3: Chunk text
            chunks = self.chunk_text(cleaned_text, chunk_config)
            
            # Step 4: Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Step 5: Create metadata
            metadata = [
                {
                    "source_file": os.path.basename(pdf_path),
                    "chunk_id": i,
                    "chunk_length": len(chunk),
                    "strategy": chunk_config.strategy
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Step 6: Store in vector database
            self.store_in_vector_db(chunks, embeddings, metadata)
            
            return {
                "status": "success",
                "total_chunks": len(chunks),
                "embedding_dimension": embeddings.shape[1],
                "chunks_preview": chunks[:3]  # First 3 chunks for preview
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"status": "error", "message": str(e)}
