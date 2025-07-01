"""
Configuration Management for Agentic AI System
==============================================

This module provides centralized configuration management using Pydantic Settings.
It demonstrates several key ML/AI concepts:

1. **Environment Variable Management**: Secure handling of API keys and sensitive data
2. **Type Safety**: Pydantic ensures all config values are properly typed
3. **Validation**: Automatic validation of configuration values
4. **Hierarchical Config**: Different settings for different environments (dev/prod)

Key AI/ML Concepts Demonstrated:
- Configuration as Code
- Secure API Key Management
- Model Parameter Tuning
- Feature Flags for A/B Testing
"""

import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """
    LLM Configuration - Demonstrates Model Parameter Management
    
    Key Concepts:
    - Temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
    - Max Tokens: Limits response length for cost control
    - Model Selection: Different models for different tasks
    """
    
    # Model Selection
    primary_model: str = Field(default="gpt-4o-mini", description="Primary LLM for game logic")
    fallback_model: str = Field(default="gpt-3.5-turbo", description="Fallback model if primary fails")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Model for creating embeddings")
    
    # Generation Parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Controls randomness in responses")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="Maximum tokens in response")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Penalty for repeating tokens")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Penalty for using new tokens")
    
    # API Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Demonstrates custom validation logic"""
        if v < 0.1:
            return 0.1  # Ensure some creativity
        return v


class MemoryConfig(BaseSettings):
    """
    Memory Configuration - Demonstrates RAG (Retrieval-Augmented Generation)
    
    Key Concepts:
    - Vector Database: Stores embeddings for semantic search
    - Chunking Strategy: How to split text for embedding
    - Retrieval Parameters: How many similar memories to retrieve
    """
    
    # Vector Database
    vector_db_path: str = Field(default="./memory_db", description="Path to vector database")
    embedding_dimension: int = Field(default=1536, description="Dimension of embeddings")
    
    # Memory Management
    max_memories: int = Field(default=10000, description="Maximum number of memories to store")
    memory_retention_days: int = Field(default=30, description="Days to keep memories")
    chunk_size: int = Field(default=512, description="Size of text chunks for embedding")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    
    # Retrieval Parameters
    top_k_retrieval: int = Field(default=5, description="Number of similar memories to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity for memory retrieval")
    
    # Memory Types
    enable_episodic_memory: bool = Field(default=True, description="Store game events")
    enable_semantic_memory: bool = Field(default=True, description="Store learned concepts")
    enable_procedural_memory: bool = Field(default=True, description="Store action patterns")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "ignore"
    }


class PlanningConfig(BaseSettings):
    """
    Planning Configuration - Demonstrates Goal-Directed Behavior
    
    Key Concepts:
    - Task Decomposition: Breaking complex goals into subtasks
    - Planning Horizon: How far ahead to plan
    - Replanning Triggers: When to update plans
    """
    
    # Planning Parameters
    planning_horizon: int = Field(default=5, description="Number of steps to plan ahead")
    max_planning_iterations: int = Field(default=3, description="Maximum planning attempts")
    replanning_threshold: float = Field(default=0.3, description="When to replan (0-1)")
    
    # Goal Management
    enable_goal_hierarchy: bool = Field(default=True, description="Use hierarchical goal structure")
    goal_priority_decay: float = Field(default=0.9, description="How quickly goals lose priority")
    
    # Task Decomposition
    max_subtasks: int = Field(default=10, description="Maximum subtasks per goal")
    subtask_timeout: int = Field(default=300, description="Timeout for subtask completion (seconds)")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "ignore"
    }


class ToolConfig(BaseSettings):
    """
    Tool Configuration - Demonstrates ReAct Pattern Implementation
    
    Key Concepts:
    - Tool Registry: Available tools for the agent
    - Tool Selection: How to choose appropriate tools
    - Tool Validation: Ensuring tool inputs are valid
    """
    
    # Tool Management
    enable_dice_rolling: bool = Field(default=True, description="Enable dice rolling tool")
    enable_combat_calculator: bool = Field(default=True, description="Enable combat calculations")
    enable_inventory_manager: bool = Field(default=True, description="Enable inventory management")
    enable_npc_generator: bool = Field(default=True, description="Enable NPC generation")
    
    # Tool Parameters
    max_tool_calls_per_turn: int = Field(default=3, description="Maximum tool calls per turn")
    tool_timeout: int = Field(default=30, description="Timeout for tool execution (seconds)")
    
    # Tool Validation
    strict_tool_validation: bool = Field(default=True, description="Validate tool inputs strictly")
    enable_tool_fallback: bool = Field(default=True, description="Use fallback when tools fail")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "ignore"
    }


class MultiAgentConfig(BaseSettings):
    """
    Multi-Agent Configuration - Demonstrates Agent Coordination
    
    Key Concepts:
    - Agent Roles: Different specialized agents
    - Communication Protocol: How agents communicate
    - Conflict Resolution: How to handle disagreements
    """
    
    # Agent Roles
    enable_dm_agent: bool = Field(default=True, description="Enable Dungeon Master agent")
    enable_npc_agents: bool = Field(default=True, description="Enable NPC agents")
    enable_player_assistant: bool = Field(default=False, description="Enable player assistance")
    
    # Communication
    agent_communication_channel: str = Field(default="shared_memory", description="How agents communicate")
    message_queue_size: int = Field(default=100, description="Size of message queue")
    
    # Coordination
    enable_consensus_mechanism: bool = Field(default=True, description="Use consensus for decisions")
    consensus_threshold: float = Field(default=0.7, description="Threshold for consensus (0-1)")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "ignore"
    }


class MonitoringConfig(BaseSettings):
    """
    Monitoring Configuration - Demonstrates Observability in AI Systems
    
    Key Concepts:
    - Metrics Collection: Tracking system performance
    - Logging Strategy: Structured logging for debugging
    - Alerting: When to notify about issues
    """
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    enable_structured_logging: bool = Field(default=True, description="Use structured logging")
    
    # Metrics
    enable_metrics_collection: bool = Field(default=True, description="Collect performance metrics")
    metrics_export_interval: int = Field(default=60, description="Metrics export interval (seconds)")
    
    # Performance Tracking
    track_response_times: bool = Field(default=True, description="Track LLM response times")
    track_token_usage: bool = Field(default=True, description="Track token consumption")
    track_memory_usage: bool = Field(default=True, description="Track memory consumption")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "ignore"
    }


class GameConfig(BaseSettings):
    """
    Game-Specific Configuration
    
    Key Concepts:
    - Game Balance: Adjusting difficulty and mechanics
    - Content Generation: How to create game content
    - Player Experience: Customizing for different players
    """
    
    # Game Balance
    difficulty_level: str = Field(default="medium", description="Game difficulty")
    combat_difficulty_multiplier: float = Field(default=1.0, description="Combat difficulty multiplier")
    experience_gain_multiplier: float = Field(default=1.0, description="Experience gain multiplier")
    
    # Content Generation
    enable_procedural_generation: bool = Field(default=True, description="Generate content procedurally")
    content_variety_factor: float = Field(default=0.8, description="Content variety (0-1)")
    
    # Player Customization
    enable_player_profiles: bool = Field(default=True, description="Store player preferences")
    adaptive_difficulty: bool = Field(default=True, description="Adjust difficulty based on player skill")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "ignore"
    }


class AgenticConfig(BaseSettings):
    """
    Main Configuration Class - Orchestrates All Sub-Configurations
    
    This demonstrates the concept of hierarchical configuration management,
    where different aspects of the system have their own specialized configs.
    """
    
    # Environment
    environment: str = Field(default="development", description="Environment (dev/prod/test)")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    
    # Sub-Configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_config()
    
    def _validate_config(self):
        """Demonstrates cross-configuration validation"""
        if self.environment == "production" and self.debug_mode:
            raise ValueError("Debug mode cannot be enabled in production")
        
        if self.llm.temperature > 1.5 and self.environment == "production":
            raise ValueError("High temperature not recommended in production")


# Global configuration instance
config = AgenticConfig()

# Configuration utilities
def get_config() -> AgenticConfig:
    """Get the global configuration instance"""
    return config

def reload_config() -> AgenticConfig:
    """Reload configuration from environment"""
    global config
    config = AgenticConfig()
    return config 