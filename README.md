# Agentic AI Dungeon Lord 🤖⚔️

A comprehensive, production-ready agentic AI system that transforms a basic D&D CLI into a fully autonomous, learning AI Dungeon Master. This project demonstrates cutting-edge AI/ML concepts in a practical, interactive gaming environment.

## 🌟 Key Features

### 🤖 **Agentic AI Architecture**
- **Perception-Action Loop**: Continuous interaction with dynamic environment
- **Multi-System Integration**: Coordinated memory, planning, and tool systems
- **Self-Reflection & Learning**: Continuous improvement from experiences
- **Goal-Directed Behavior**: Hierarchical planning and task decomposition

### 🧠 **Advanced Memory System (RAG)**
- **Vector Database Integration**: ChromaDB for semantic memory storage
- **Memory Types**: Episodic, semantic, procedural, emotional, and spatial
- **Memory Consolidation**: Short-term to long-term memory conversion
- **Context Window Optimization**: Efficient token management for LLMs

### 🎯 **Intelligent Planning System**
- **Hierarchical Task Networks (HTN)**: Complex goal decomposition
- **Dynamic Replanning**: Adaptive planning based on changing circumstances
- **Goal Management**: Priority-based goal hierarchy and conflict resolution
- **Task Execution**: Automated task completion with outcome tracking

### 🔧 **Tool Integration (ReAct Pattern)**
- **Dice Rolling**: D&D-compliant random number generation
- **Combat Calculator**: Automated combat resolution
- **Inventory Manager**: Item and equipment tracking
- **NPC Generator**: Dynamic character creation
- **Tool Validation**: Input sanitization and error handling

### 📊 **Comprehensive Monitoring**
- **Performance Metrics**: Real-time system performance tracking
- **Structured Logging**: JSON-formatted logs for analysis
- **Memory Analytics**: Usage patterns and optimization insights
- **Tool Effectiveness**: Success rates and execution times

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- 4GB+ RAM (for vector database)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cli-dungeonlord
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp env_example.txt .env
   # Edit .env with your OpenAI API key
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## 🎮 Usage

### Interactive Commands
- **Natural Language**: Type as you would with a human DM
- **System Commands**: `help`, `status`, `stats`, `tools`, `memory`, `planning`, `demo`
- **Exit**: `quit`, `exit`, or `q`

### Example Interactions
```
You: I want to explore the dungeon
🤖 AI: *analyzes request, creates exploration goal, generates plan*

You: Roll a d20 for perception
🤖 AI: *uses dice tool, incorporates result into narrative*

You: I encounter a goblin, what do I do?
🤖 AI: *retrieves combat memories, uses combat calculator, provides tactical advice*

You: Generate an NPC merchant
🤖 AI: *uses NPC generator tool, creates detailed character with personality*
```

## 🧠 AI/ML Concepts Demonstrated

### 1. **RAG (Retrieval-Augmented Generation)**
- Vector embeddings for semantic search
- Memory retrieval based on context relevance
- Dynamic context window management

### 2. **ReAct Pattern (Reasoning + Acting)**
- Interleaved reasoning and tool execution
- Structured tool calling with validation
- Result integration into responses

### 3. **Hierarchical Planning**
- Goal decomposition into actionable tasks
- Dependency resolution and scheduling
- Dynamic replanning based on outcomes

### 4. **Memory Management**
- Multiple memory types (episodic, semantic, procedural)
- Memory consolidation and forgetting mechanisms
- Access pattern optimization

### 5. **Multi-Agent Coordination**
- Specialized agent roles (DM, NPCs, Player Assistant)
- Communication protocols and consensus mechanisms
- Conflict resolution strategies

### 6. **Continuous Learning**
- Self-reflection and performance analysis
- Behavior adaptation based on outcomes
- Memory-based learning from interactions

## 📁 Project Structure

```
cli-dungeonlord/
├── main.py                 # Main application entry point
├── agentic_agent.py        # Core agentic AI implementation
├── memory_system.py        # RAG-based memory management
├── planning_system.py      # Goal-directed planning
├── tools_system.py         # Tool integration and execution
├── config.py              # Configuration management
├── game_state.py          # Game state tracking
├── prompts.py             # LLM prompt templates
├── requirements.txt       # Python dependencies
├── env_example.txt        # Environment configuration template
└── README.md             # This file
```

## 🔧 Configuration

The system is highly configurable through environment variables:

### Core Settings
- `OPENAI_API_KEY`: Your OpenAI API key
- `ENVIRONMENT`: development/production/test
- `DEBUG_MODE`: Enable debug features

### LLM Configuration
- `PRIMARY_MODEL`: Main LLM model (gpt-4o-mini, gpt-4o, etc.)
- `TEMPERATURE`: Response creativity (0.0-2.0)
- `MAX_TOKENS`: Response length limit

### Memory Settings
- `VECTOR_DB_PATH`: Vector database location
- `MAX_MEMORIES`: Maximum stored memories
- `SIMILARITY_THRESHOLD`: Memory retrieval threshold

### Planning Settings
- `PLANNING_HORIZON`: How far ahead to plan
- `REPLANNING_THRESHOLD`: When to replan
- `MAX_SUBTASKS`: Maximum subtasks per goal

## 📊 Performance Monitoring

### Real-time Metrics
- Response times and throughput
- Memory hit rates and utilization
- Tool success rates and execution times
- Goal completion rates

### System Health
- Memory usage and optimization
- Planning efficiency and success rates
- Tool availability and performance
- Overall agent effectiveness

## 🎯 Learning Objectives

This project serves as a comprehensive learning platform for:

### AI/ML Fundamentals
- **LLM Integration**: Working with large language models
- **Vector Databases**: Semantic search and similarity
- **Prompt Engineering**: Effective LLM communication
- **System Architecture**: Scalable AI system design

### Advanced Concepts
- **Agentic AI**: Goal-directed autonomous systems
- **Multi-Agent Systems**: Coordination and communication
- **Memory Systems**: Cognitive architecture design
- **Tool Integration**: External API and service integration

### Production Skills
- **Configuration Management**: Environment-based settings
- **Monitoring & Observability**: Performance tracking
- **Error Handling**: Graceful failure management
- **Testing & Validation**: System reliability

## 🔮 Future Enhancements

### Planned Features
- **Multi-Player Support**: Multiple players and party management
- **Advanced Combat**: Turn-based combat with initiative
- **Quest System**: Dynamic quest generation and tracking
- **World Generation**: Procedural dungeon and world creation
- **Voice Integration**: Speech-to-text and text-to-speech
- **Web Interface**: Browser-based UI

### Research Areas
- **Multi-Modal AI**: Image and audio processing
- **Reinforcement Learning**: Adaptive difficulty and behavior
- **Federated Learning**: Distributed AI training
- **Quantum Computing**: Quantum-enhanced AI algorithms

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models
- ChromaDB for vector database technology
- The D&D community for inspiration
- All contributors and testers

---

**Ready to experience the future of AI-powered gaming? Start your adventure with Agentic AI Dungeon Lord!** 🚀⚔️
