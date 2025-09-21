"""RAG retrieval system with TF-IDF and document processing."""

import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A document chunk with metadata."""
    id: str
    content: str
    source: str
    chunk_index: int
    score: float = 0.0


class RAGGraphDistiller:
    """Creates compact graph representation of documents based on requirements."""
    
    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)
        self.documents = {}
        self.chunks = []
        self._load_documents()
    
    def _load_documents(self):
        """Load all documents from the docs directory."""
        for doc_file in self.docs_dir.glob("*.md"):
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
                self.documents[doc_file.name] = content
                self._chunk_document(doc_file.name, content)
    
    def _chunk_document(self, filename: str, content: str, chunk_size: int = 300):
        """Split document into chunks."""
        # Split by paragraphs first, then by sentences if needed
        paragraphs = content.split('\n\n')
        
        chunk_index = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is too long, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk + sentence) > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_id = f"{filename.replace('.md', '')}::chunk{chunk_index}"
                        self.chunks.append(DocumentChunk(
                            id=chunk_id,
                            content=current_chunk.strip(),
                            source=filename,
                            chunk_index=chunk_index
                        ))
                        chunk_index += 1
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence if current_chunk else sentence
                
                # Save remaining chunk
                if current_chunk:
                    chunk_id = f"{filename.replace('.md', '')}::chunk{chunk_index}"
                    self.chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=current_chunk.strip(),
                        source=filename,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1
            else:
                # Paragraph fits in one chunk
                chunk_id = f"{filename.replace('.md', '')}::chunk{chunk_index}"
                self.chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=para,
                    source=filename,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
    
    def create_graph_representation(self, ontological_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a compact graph representation based on requirements."""
        relevant_chunks = []
        keywords = set()
        
        # Extract keywords from ontological blocks
        for block in ontological_blocks:
            content = str(block.get('content', ''))
            # Extract meaningful keywords
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            keywords.update(words)
        
        # Filter chunks based on keyword relevance
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', chunk.content.lower()))
            if keywords.intersection(chunk_words):
                relevant_chunks.append(chunk)
        
        return {
            "relevant_chunks": relevant_chunks,
            "keywords": list(keywords),
            "total_chunks": len(self.chunks),
            "relevant_count": len(relevant_chunks)
        }


class RAGRetriever:
    """TF-IDF based document retriever."""
    
    def __init__(self, docs_dir: str):
        self.distiller = RAGGraphDistiller(docs_dir)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.chunk_vectors = None
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index for all chunks."""
        if not self.distiller.chunks:
            return
        
        chunk_texts = [chunk.content for chunk in self.distiller.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
    
    def retrieve(self, query: str, top_k: int = 5, 
                ontological_blocks: Optional[List[Dict[str, Any]]] = None) -> List[DocumentChunk]:
        """Retrieve top-k most relevant chunks."""
        if not self.chunk_vectors or not self.distiller.chunks:
            return []
        
        # First, create graph representation to narrow down search space
        if ontological_blocks:
            graph_repr = self.distiller.create_graph_representation(ontological_blocks)
            relevant_chunks = graph_repr["relevant_chunks"]
            if relevant_chunks:
                # Use only relevant chunks for search
                search_chunks = relevant_chunks
                search_texts = [chunk.content for chunk in search_chunks]
                if search_texts:
                    search_vectors = self.vectorizer.transform(search_texts)
                else:
                    search_chunks = self.distiller.chunks
                    search_vectors = self.chunk_vectors
            else:
                search_chunks = self.distiller.chunks
                search_vectors = self.chunk_vectors
        else:
            search_chunks = self.distiller.chunks
            search_vectors = self.chunk_vectors
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, search_vectors).flatten()
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Create result chunks with scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include chunks with positive similarity
                chunk = search_chunks[idx]
                chunk.score = float(similarities[idx])
                results.append(chunk)
        
        return results
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all document chunks."""
        return self.distiller.chunks
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a specific chunk by ID."""
        for chunk in self.distiller.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None


class RAGGraphDistillerNode:
    """LangGraph node for RAG graph distillation."""
    
    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and create graph representation."""
        ontological_blocks = state.get("ontological_blocks", [])
        print("Node start: RAGGraphDistiller")
        
        # Create graph representation
        graph_repr = self.retriever.distiller.create_graph_representation(ontological_blocks)
        
        # Generate potential search tags
        keywords = graph_repr.get("keywords", [])
        search_tags = []
        
        # Add semantic tags based on keywords
        if any(word in keywords for word in ["return", "policy", "day"]):
            search_tags.append("return_policy")
        if any(word in keywords for word in ["marketing", "campaign", "date"]):
            search_tags.append("marketing_calendar")
        if any(word in keywords for word in ["kpi", "aov", "margin", "revenue"]):
            search_tags.append("kpi_definitions")
        if any(word in keywords for word in ["category", "product", "catalog"]):
            search_tags.append("catalog")
        
        result_state = {
            **state,
            "rag_graph_repr": graph_repr,
            "rag_search_tags": search_tags
        }
        print("Node end: RAGGraphDistiller")
        return result_state


class RAGRetrieverNode:
    """LangGraph node for RAG retrieval."""
    
    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant document chunks."""
        question = state.get("question", "")
        ontological_blocks = state.get("ontological_blocks", [])
        top_k = state.get("rag_top_k", 5)
        print("Node start: RAGRetriever")
        
        # Retrieve chunks
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            ontological_blocks=ontological_blocks
        )
        
        # Convert to dictionaries for state
        chunks_data = []
        for chunk in retrieved_chunks:
            chunks_data.append({
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "score": chunk.score
            })
        
        result_state = {
            **state,
            "rag_chunks": chunks_data,
            "rag_chunk_count": len(chunks_data)
        }
        print(f"Node end: RAGRetriever -> chunks={len(chunks_data)}")
        return result_state
