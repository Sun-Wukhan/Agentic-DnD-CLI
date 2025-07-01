"""
Memory System for Agentic AI
============================

This module implements a sophisticated memory system that demonstrates key AI/ML concepts:

1. **RAG (Retrieval-Augmented Generation)**: Using vector embeddings for semantic search
2. **Memory Types**: Episodic, semantic, and procedural memory
3. **Memory Consolidation**: Converting short-term to long-term memory
4. **Context Window Management**: Efficient use of token limits

Key AI/ML Concepts Demonstrated:
- Vector Embeddings and Similarity Search
- Memory Consolidation and Forgetting
- Context Window Optimization
- Semantic Memory Organization
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import structlog

from config import get_config

logger = structlog.get_logger(__name__)


class MemoryType(Enum):
    """Types of memory - demonstrates different cognitive functions"""
    EPISODIC = "episodic"      # Events and experiences
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    EMOTIONAL = "emotional"    # Emotional associations
    SPATIAL = "spatial"        # Location and navigation


class MemoryPriority(Enum):
    """Memory priority levels - demonstrates importance weighting"""
    CRITICAL = 5    # Essential for survival/functioning
    HIGH = 4        # Important for current goals
    MEDIUM = 3      # Useful but not critical
    LOW = 2         # Nice to have
    TRIVIAL = 1     # Can be forgotten


@dataclass
class Memory:
    """
    Memory Data Structure - Demonstrates Structured Data Management
    
    Key Concepts:
    - Unique Identification: Each memory has a unique ID
    - Temporal Information: When the memory was created/accessed
    - Semantic Content: The actual information stored
    - Metadata: Additional context and properties
    - Access Patterns: How often and when the memory is accessed
    """
    
    id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    created_at: datetime
    last_accessed: datetime
    access_count: int
    emotional_valence: float  # -1.0 (negative) to 1.0 (positive)
    confidence: float  # 0.0 to 1.0
    tags: List[str]
    source: str
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "emotional_valence": self.emotional_valence,
            "confidence": self.confidence,
            "tags": self.tags,
            "source": self.source,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create memory from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            priority=MemoryPriority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data["access_count"],
            emotional_valence=data["emotional_valence"],
            confidence=data["confidence"],
            tags=data["tags"],
            source=data["source"],
            context=data["context"]
        )


class MemoryManager:
    """
    Memory Manager - Orchestrates All Memory Operations
    
    This class demonstrates several key AI/ML concepts:
    
    1. **Vector Database Integration**: Using ChromaDB for semantic search
    2. **Memory Consolidation**: Converting short-term to long-term memory
    3. **Forgetting Mechanisms**: Removing less important memories
    4. **Context Window Management**: Optimizing memory retrieval for LLM context
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(component="memory_manager")
        
        # Initialize vector database
        self._init_vector_db()
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Short-term memory (recent events)
        self.short_term_memory: List[Memory] = []
        
        # Memory consolidation queue
        self.consolidation_queue: List[Memory] = []
        
        # Memory access statistics
        self.access_stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "average_similarity": 0.0,
            "memory_hit_rate": 0.0
        }
    
    def _init_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.memory.vector_db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create collections for different memory types
            self.collections = {}
            for memory_type in MemoryType:
                collection_name = f"memory_{memory_type.value}"
                self.collections[memory_type] = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": f"Collection for {memory_type.value} memories"}
                )
            
            self.logger.info("Vector database initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize vector database", error=str(e))
            raise
    
    def _init_embedding_model(self):
        """Initialize sentence transformer for embeddings"""
        try:
            # Use a smaller model for efficiency in development
            model_name = "all-MiniLM-L6-v2" if self.config.environment == "development" else "all-mpnet-base-v2"
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"Embedding model initialized: {model_name}")
            
        except Exception as e:
            self.logger.error("Failed to initialize embedding model", error=str(e))
            raise
    
    def create_memory(
        self,
        content: str,
        memory_type: MemoryType,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        emotional_valence: float = 0.0,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        source: str = "user_input",
        context: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Create a new memory - Demonstrates Memory Formation
        
        Key Concepts:
        - Memory Encoding: Converting information into storable format
        - Priority Assignment: Determining importance of information
        - Context Preservation: Maintaining relevant context
        """
        
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            priority=priority,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            emotional_valence=emotional_valence,
            confidence=confidence,
            tags=tags or [],
            source=source,
            context=context or {}
        )
        
        # Add to short-term memory first
        self.short_term_memory.append(memory)
        
        # If high priority, add to consolidation queue
        if priority.value >= MemoryPriority.HIGH.value:
            self.consolidation_queue.append(memory)
        
        self.logger.info(
            "Memory created",
            memory_id=memory.id,
            memory_type=memory_type.value,
            priority=priority.value,
            content_length=len(content)
        )
        
        return memory
    
    def store_memory(self, memory: Memory) -> bool:
        """
        Store memory in vector database - Demonstrates Long-term Memory Storage
        
        Key Concepts:
        - Vector Embedding: Converting text to numerical representation
        - Semantic Indexing: Enabling similarity-based retrieval
        - Metadata Storage: Preserving memory properties
        """
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(memory.content).tolist()
            
            # Store in appropriate collection
            collection = self.collections[memory.memory_type]
            collection.add(
                embeddings=[embedding],
                documents=[memory.content],
                metadatas=[memory.to_dict()],
                ids=[memory.id]
            )
            
            self.logger.info(
                "Memory stored in vector database",
                memory_id=memory.id,
                memory_type=memory.memory_type.value
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to store memory",
                memory_id=memory.id,
                error=str(e)
            )
            return False
    
    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_age_days: Optional[int] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve relevant memories - Demonstrates RAG Implementation
        
        Key Concepts:
        - Semantic Search: Finding memories based on meaning, not just keywords
        - Multi-Collection Search: Searching across different memory types
        - Relevance Scoring: Ranking memories by similarity
        - Temporal Filtering: Considering memory age
        """
        
        if memory_types is None:
            memory_types = list(MemoryType)
        
        if top_k is None:
            top_k = self.config.memory.top_k_retrieval
        
        if similarity_threshold is None:
            similarity_threshold = self.config.memory.similarity_threshold
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        # Search in each memory type collection
        for memory_type in memory_types:
            if memory_type not in self.collections:
                continue
            
            collection = self.collections[memory_type]
            
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )
                
                # Process results
                for i, (metadata, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    similarity = 1 - distance
                    
                    if similarity >= similarity_threshold:
                        memory = Memory.from_dict(metadata)
                        
                        # Apply temporal filter
                        if max_age_days:
                            age = datetime.now() - memory.created_at
                            if age.days > max_age_days:
                                continue
                        
                        all_results.append((memory, similarity))
                        
            except Exception as e:
                self.logger.error(
                    "Error retrieving from collection",
                    memory_type=memory_type.value,
                    error=str(e)
                )
        
        # Sort by similarity and update access statistics
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Update access statistics
        self._update_access_stats(all_results)
        
        # Update memory access patterns
        for memory, _ in all_results:
            memory.last_accessed = datetime.now()
            memory.access_count += 1
        
        self.logger.info(
            "Memory retrieval completed",
            query_length=len(query),
            results_count=len(all_results),
            memory_types=[mt.value for mt in memory_types]
        )
        
        return all_results
    
    def _update_access_stats(self, results: List[Tuple[Memory, float]]):
        """Update memory access statistics"""
        self.access_stats["total_retrievals"] += 1
        
        if results:
            self.access_stats["successful_retrievals"] += 1
            similarities = [sim for _, sim in results]
            self.access_stats["average_similarity"] = np.mean(similarities)
            self.access_stats["memory_hit_rate"] = (
                self.access_stats["successful_retrievals"] / 
                self.access_stats["total_retrievals"]
            )
    
    def consolidate_memories(self) -> int:
        """
        Consolidate short-term memories to long-term storage
        
        Key Concepts:
        - Memory Consolidation: Converting temporary to permanent storage
        - Priority-based Selection: Important memories are consolidated first
        - Batch Processing: Efficient handling of multiple memories
        """
        
        consolidated_count = 0
        
        # Process consolidation queue
        while self.consolidation_queue:
            memory = self.consolidation_queue.pop(0)
            
            if self.store_memory(memory):
                consolidated_count += 1
                # Remove from short-term memory
                self.short_term_memory = [m for m in self.short_term_memory if m.id != memory.id]
        
        # Consolidate some short-term memories based on priority and age
        current_time = datetime.now()
        memories_to_consolidate = []
        
        for memory in self.short_term_memory:
            age_hours = (current_time - memory.created_at).total_seconds() / 3600
            
            # Consolidate if old enough or high priority
            if (age_hours > 24 or 
                memory.priority.value >= MemoryPriority.HIGH.value or
                memory.access_count > 5):
                memories_to_consolidate.append(memory)
        
        # Store memories in batches
        for memory in memories_to_consolidate:
            if self.store_memory(memory):
                consolidated_count += 1
                self.short_term_memory.remove(memory)
        
        self.logger.info(
            "Memory consolidation completed",
            consolidated_count=consolidated_count,
            short_term_count=len(self.short_term_memory)
        )
        
        return consolidated_count
    
    def forget_memories(self, max_memories: Optional[int] = None) -> int:
        """
        Remove less important memories - Demonstrates Forgetting Mechanisms
        
        Key Concepts:
        - Memory Decay: Less important memories are forgotten
        - Access-based Retention: Frequently accessed memories are kept
        - Priority-based Deletion: Critical memories are preserved
        """
        
        if max_memories is None:
            max_memories = self.config.memory.max_memories
        
        forgotten_count = 0
        
        # Get all memories from all collections
        all_memories = []
        for memory_type, collection in self.collections.items():
            try:
                results = collection.get()
                for i, metadata in enumerate(results["metadatas"]):
                    memory = Memory.from_dict(metadata)
                    all_memories.append((memory, memory_type, results["ids"][i]))
            except Exception as e:
                self.logger.error(f"Error getting memories from {memory_type.value}", error=str(e))
        
        # Sort by importance score (priority * access_count * age_factor)
        def importance_score(memory_tuple):
            memory, _, _ = memory_tuple
            age_days = (datetime.now() - memory.created_at).days
            age_factor = max(0.1, 1.0 - (age_days / 365))  # Decay over time
            return memory.priority.value * memory.access_count * age_factor
        
        all_memories.sort(key=importance_score)
        
        # Remove least important memories if we exceed the limit
        if len(all_memories) > max_memories:
            memories_to_remove = all_memories[:len(all_memories) - max_memories]
            
            for memory, memory_type, memory_id in memories_to_remove:
                # Don't remove critical memories
                if memory.priority == MemoryPriority.CRITICAL:
                    continue
                
                try:
                    collection = self.collections[memory_type]
                    collection.delete(ids=[memory_id])
                    forgotten_count += 1
                    
                    self.logger.info(
                        "Memory forgotten",
                        memory_id=memory_id,
                        memory_type=memory_type.value,
                        priority=memory.priority.value
                    )
                    
                except Exception as e:
                    self.logger.error(
                        "Failed to forget memory",
                        memory_id=memory_id,
                        error=str(e)
                    )
        
        return forgotten_count
    
    def get_context_window(
        self,
        query: str,
        max_tokens: int = 2000,
        include_recent: bool = True
    ) -> str:
        """
        Get optimized context window for LLM - Demonstrates Context Management
        
        Key Concepts:
        - Token Budget Management: Efficient use of context window
        - Relevance Prioritization: Most relevant memories first
        - Recency Bias: Recent events are important
        - Context Compression: Summarizing when needed
        """
        
        # Estimate tokens per character (rough approximation)
        chars_per_token = 4
        
        # Calculate available space
        available_chars = max_tokens * chars_per_token
        
        context_parts = []
        current_length = 0
        
        # Add recent memories if requested
        if include_recent and self.short_term_memory:
            recent_memories = sorted(
                self.short_term_memory,
                key=lambda m: m.last_accessed,
                reverse=True
            )[:3]  # Last 3 recent memories
            
            for memory in recent_memories:
                memory_text = f"[RECENT] {memory.content}"
                if current_length + len(memory_text) < available_chars * 0.3:  # Use 30% for recent
                    context_parts.append(memory_text)
                    current_length += len(memory_text)
        
        # Retrieve relevant long-term memories
        relevant_memories = self.retrieve_memories(
            query=query,
            top_k=10,
            similarity_threshold=0.6
        )
        
        # Add relevant memories
        for memory, similarity in relevant_memories:
            memory_text = f"[{memory.memory_type.value.upper()}] {memory.content} (relevance: {similarity:.2f})"
            
            if current_length + len(memory_text) < available_chars * 0.7:  # Use 70% for relevant
                context_parts.append(memory_text)
                current_length += len(memory_text)
            else:
                break
        
        return "\n".join(context_parts)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            "short_term_count": len(self.short_term_memory),
            "consolidation_queue_size": len(self.consolidation_queue),
            "access_stats": self.access_stats.copy(),
            "collections": {}
        }
        
        # Get collection statistics
        for memory_type, collection in self.collections.items():
            try:
                count = collection.count()
                stats["collections"][memory_type.value] = count
            except Exception as e:
                self.logger.error(f"Error getting count for {memory_type.value}", error=str(e))
                stats["collections"][memory_type.value] = 0
        
        return stats


# Global memory manager instance
memory_manager = MemoryManager()


# Utility functions for easy access
def create_memory(*args, **kwargs) -> Memory:
    """Create a new memory"""
    return memory_manager.create_memory(*args, **kwargs)


def retrieve_memories(*args, **kwargs) -> List[Tuple[Memory, float]]:
    """Retrieve relevant memories"""
    return memory_manager.retrieve_memories(*args, **kwargs)


def get_context_window(*args, **kwargs) -> str:
    """Get optimized context window"""
    return memory_manager.get_context_window(*args, **kwargs)


def get_memory_stats() -> Dict[str, Any]:
    """Get memory statistics"""
    return memory_manager.get_memory_stats() 