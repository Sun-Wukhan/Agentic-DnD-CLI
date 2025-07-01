"""
Agentic Agent - Main AI Agent Implementation
============================================

This module implements the main agentic AI agent that orchestrates all subsystems.
It demonstrates the complete agentic AI architecture with:

1. **Perception-Action Loop**: Continuous interaction with the environment
2. **Multi-System Integration**: Coordinating memory, planning, and tools
3. **ReAct Pattern**: Reasoning and Acting in interleaved cycles
4. **Self-Reflection**: Learning and improving from experiences

Key AI/ML Concepts Demonstrated:
- Agent Architecture Design
- System Integration and Coordination
- Continuous Learning and Adaptation
- Performance Monitoring and Optimization
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from openai import OpenAI

from config import get_config
from memory_system import (
    MemoryManager, create_memory, retrieve_memories, get_context_window,
    MemoryType, MemoryPriority
)
from planning_system import (
    PlanningSystem, create_goal, create_task, generate_plan,
    GoalType, GoalStatus, TaskStatus
)
from tools_system import (
    ToolsSystem, execute_tool, select_tool, get_tool_stats,
    ToolCall, ToolStatus
)
from game_state import GameState

logger = structlog.get_logger(__name__)
console = Console()


class AgentState(Enum):
    """Agent state enumeration"""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    LEARNING = "learning"


class AgentMode(Enum):
    """Agent operating modes"""
    REACTIVE = "reactive"      # Immediate responses
    PLANNING = "planning"      # Goal-directed planning
    REFLECTIVE = "reflective"  # Self-reflection and learning
    AUTONOMOUS = "autonomous"  # Full autonomy


@dataclass
class AgentContext:
    """
    Agent Context - Maintains Current State and Information
    
    Key Concepts:
    - Current Situation: What's happening right now
    - Active Goals: What the agent is trying to achieve
    - Available Resources: What the agent can use
    - Environmental State: Current game/world state
    """
    
    current_situation: Dict[str, Any] = field(default_factory=dict)
    active_goals: List[str] = field(default_factory=list)
    available_resources: Dict[str, Any] = field(default_factory=dict)
    environmental_state: Dict[str, Any] = field(default_factory=dict)
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class AgenticAgent:
    """
    Agentic Agent - Main AI Agent Implementation
    
    This class demonstrates the complete agentic AI architecture:
    
    1. **System Integration**: Coordinating memory, planning, and tools
    2. **Perception-Action Loop**: Continuous interaction with environment
    3. **Learning and Adaptation**: Improving from experiences
    4. **Performance Optimization**: Monitoring and enhancing capabilities
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(component="agentic_agent")
        
        # Initialize subsystems
        self.memory_manager = MemoryManager()
        self.planning_system = PlanningSystem()
        self.tools_system = ToolsSystem()
        
        # Initialize LLM client
        self.llm_client = OpenAI(api_key=self.config.llm.openai_api_key)
        
        # Agent state
        self.state = AgentState.IDLE
        self.mode = AgentMode.PLANNING
        self.context = AgentContext()
        
        # Performance tracking
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_actions": 0,
            "planning_cycles": 0,
            "reflection_cycles": 0,
            "average_response_time": 0.0,
            "goal_completion_rate": 0.0,
            "memory_utilization": 0.0,
            "tool_effectiveness": 0.0
        }
        
        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Initialize agent
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent with basic knowledge and capabilities"""
        
        # Create initial memories
        self._create_initial_memories()
        
        # Set up initial goals
        self._setup_initial_goals()
        
        # Initialize game state
        self.game_state = GameState()
        
        self.logger.info("Agentic agent initialized successfully")
    
    def _create_initial_memories(self):
        """Create initial semantic memories for the agent"""
        
        initial_memories = [
            {
                "content": "I am a Dungeon Master for a D&D 5th Edition game. I must create engaging, fair, and immersive experiences for players.",
                "memory_type": MemoryType.SEMANTIC,
                "priority": MemoryPriority.CRITICAL,
                "tags": ["role", "identity", "dungeon_master"]
            },
            {
                "content": "D&D 5th Edition uses a d20 system for most checks. Roll d20 + modifiers vs target number (DC or AC).",
                "memory_type": MemoryType.SEMANTIC,
                "priority": MemoryPriority.HIGH,
                "tags": ["rules", "d20_system", "mechanics"]
            },
            {
                "content": "Combat follows initiative order. Each creature gets one action, one bonus action, and one reaction per turn.",
                "memory_type": MemoryType.SEMANTIC,
                "priority": MemoryPriority.HIGH,
                "tags": ["rules", "combat", "initiative"]
            },
            {
                "content": "Always maintain game balance. Challenges should be appropriate for player level and party composition.",
                "memory_type": MemoryType.SEMANTIC,
                "priority": MemoryPriority.HIGH,
                "tags": ["balance", "challenge", "game_design"]
            },
            {
                "content": "Narrative should be vivid and engaging. Describe environments, actions, and consequences in detail.",
                "memory_type": MemoryType.SEMANTIC,
                "priority": MemoryPriority.MEDIUM,
                "tags": ["narrative", "storytelling", "description"]
            }
        ]
        
        for memory_data in initial_memories:
            create_memory(**memory_data)
        
        self.logger.info(f"Created {len(initial_memories)} initial memories")
    
    def _setup_initial_goals(self):
        """Set up initial goals for the agent"""
        
        # Main goal: Create engaging D&D experience
        main_goal = create_goal(
            name="Create Engaging D&D Experience",
            description="Provide an immersive, fair, and enjoyable D&D 5th Edition experience for players",
            goal_type=GoalType.ACHIEVEMENT,
            priority=10,
            success_criteria=[
                "Players are engaged and having fun",
                "Game follows D&D 5e rules correctly",
                "Challenges are appropriate for party level",
                "Narrative is immersive and coherent"
            ]
        )
        
        # Sub-goals
        sub_goals = [
            {
                "name": "Maintain Game Balance",
                "description": "Ensure challenges and rewards are appropriate for the party",
                "goal_type": GoalType.MAINTENANCE,
                "priority": 8
            },
            {
                "name": "Enforce Rules Fairly",
                "description": "Apply D&D 5e rules consistently and fairly",
                "goal_type": GoalType.MAINTENANCE,
                "priority": 9
            },
            {
                "name": "Create Immersive Narrative",
                "description": "Provide vivid descriptions and engaging storytelling",
                "goal_type": GoalType.ACHIEVEMENT,
                "priority": 7
            }
        ]
        
        for sub_goal_data in sub_goals:
            sub_goal = create_goal(
                parent_goal=main_goal.id,
                **sub_goal_data
            )
            main_goal.sub_goals.append(sub_goal.id)
        
        self.context.active_goals = [main_goal.id]
        
        self.logger.info("Initial goals set up successfully")
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response - Main Perception-Action Loop
        
        Key Concepts:
        - Input Processing: Understanding user intent
        - Context Retrieval: Getting relevant information
        - Planning: Determining what to do
        - Action Execution: Carrying out planned actions
        - Response Generation: Creating appropriate response
        """
        
        start_time = time.time()
        self.state = AgentState.THINKING
        
        try:
            # Update context
            self._update_context(user_input)
            
            # Retrieve relevant memories
            context_window = self._get_relevant_context(user_input)
            
            # Analyze input and determine response strategy
            response_strategy = self._analyze_input(user_input, context_window)
            
            # Execute response strategy
            response = self._execute_response_strategy(response_strategy, user_input)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, response_strategy)
            
            # Store interaction
            self._store_interaction(user_input, response, response_strategy)
            
            # Periodic reflection and learning
            if self.performance_metrics["total_interactions"] % 10 == 0:
                self._reflect_and_learn()
            
            return response
            
        except Exception as e:
            self.logger.error("Error processing input", error=str(e))
            self.state = AgentState.IDLE
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _update_context(self, user_input: str):
        """Update agent context with current situation"""
        
        # Update current situation
        self.context.current_situation.update({
            "last_input": user_input,
            "timestamp": datetime.now().isoformat(),
            "input_length": len(user_input),
            "input_type": self._classify_input(user_input)
        })
        
        # Update environmental state from game state
        self.context.environmental_state.update({
            "location": self.game_state.location,
            "party_status": self.game_state.party,
            "game_history": len(self.game_state.history)
        })
    
    def _classify_input(self, user_input: str) -> str:
        """Classify the type of user input"""
        input_lower = user_input.lower()
        
        if any(word in input_lower for word in ["attack", "fight", "combat"]):
            return "combat_action"
        elif any(word in input_lower for word in ["look", "examine", "search"]):
            return "exploration_action"
        elif any(word in input_lower for word in ["talk", "speak", "ask"]):
            return "social_action"
        elif any(word in input_lower for word in ["move", "go", "walk"]):
            return "movement_action"
        elif any(word in input_lower for word in ["inventory", "equipment", "item"]):
            return "inventory_action"
        else:
            return "general_action"
    
    def _get_relevant_context(self, user_input: str) -> str:
        """Get relevant context for processing input"""
        
        # Get memory context
        memory_context = get_context_window(
            query=user_input,
            max_tokens=1500,
            include_recent=True
        )
        
        # Get current game state
        game_context = f"""
Current Location: {self.game_state.location}
Party Status: {', '.join(f'{name} (HP: {attrs["HP"]})' for name, attrs in self.game_state.party.items())}
Recent History: {len(self.game_state.history)} events recorded
        """.strip()
        
        # Get active goals context
        goals_context = ""
        if self.context.active_goals:
            active_goals = [self.planning_system.goals[goal_id] for goal_id in self.context.active_goals if goal_id in self.planning_system.goals]
            if active_goals:
                goals_context = "Active Goals:\n" + "\n".join(
                    f"- {goal.name} (Progress: {goal.progress:.1%})"
                    for goal in active_goals
                )
        
        return f"""
{memory_context}

{game_context}

{goals_context}
        """.strip()
    
    def _analyze_input(self, user_input: str, context: str) -> Dict[str, Any]:
        """
        Analyze user input and determine response strategy
        
        Key Concepts:
        - Intent Recognition: Understanding what the user wants
        - Context Integration: Using relevant information
        - Strategy Selection: Choosing appropriate response approach
        - Tool Identification: Determining if tools are needed
        """
        
        # Use LLM to analyze input and determine strategy
        analysis_prompt = f"""
You are an intelligent D&D Dungeon Master agent. Analyze the user's input and determine the best response strategy.

User Input: {user_input}

Context:
{context}

Available Tools: {', '.join(tool['name'] for tool in self.tools_system.list_tools())}

Determine the response strategy. Return a JSON object with:
{{
    "strategy": "immediate_response|tool_use|planning|reflection",
    "reasoning": "Why this strategy is appropriate",
    "required_tools": ["list", "of", "tools", "needed"],
    "response_focus": "What the response should focus on",
    "complexity": "simple|moderate|complex"
}}

Analysis:
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.primary_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            self.logger.error("Error analyzing input", error=str(e))
            # Fallback to simple strategy
            return {
                "strategy": "immediate_response",
                "reasoning": "Fallback due to analysis error",
                "required_tools": [],
                "response_focus": "Provide immediate response",
                "complexity": "simple"
            }
    
    def _execute_response_strategy(self, strategy: Dict[str, Any], user_input: str) -> str:
        """
        Execute the determined response strategy
        
        Key Concepts:
        - Strategy Execution: Carrying out planned approach
        - Tool Integration: Using tools when needed
        - Response Generation: Creating appropriate output
        - Quality Assurance: Ensuring response quality
        """
        
        strategy_type = strategy.get("strategy", "immediate_response")
        
        if strategy_type == "tool_use":
            return self._execute_tool_based_response(strategy, user_input)
        elif strategy_type == "planning":
            return self._execute_planning_response(strategy, user_input)
        elif strategy_type == "reflection":
            return self._execute_reflection_response(strategy, user_input)
        else:
            return self._execute_immediate_response(strategy, user_input)
    
    def _execute_tool_based_response(self, strategy: Dict[str, Any], user_input: str) -> str:
        """Execute response using tools"""
        
        self.state = AgentState.EXECUTING
        
        # Select and execute tools
        tool_results = []
        required_tools = strategy.get("required_tools", [])
        
        for tool_name in required_tools:
            # Determine tool parameters based on input
            tool_params = self._determine_tool_parameters(tool_name, user_input)
            
            if tool_params:
                tool_call = execute_tool(tool_name, tool_params)
                tool_results.append(tool_call)
        
        # Generate response incorporating tool results
        response = self._generate_tool_response(user_input, tool_results, strategy)
        
        self.state = AgentState.IDLE
        return response
    
    def _determine_tool_parameters(self, tool_name: str, user_input: str) -> Optional[Dict[str, Any]]:
        """Determine parameters for tool execution"""
        
        if tool_name == "roll_dice":
            # Extract dice information from input
            import re
            dice_pattern = r'(\d+)d(\d+)(?:\+(\d+))?'
            match = re.search(dice_pattern, user_input.lower())
            
            if match:
                count = int(match.group(1))
                sides = int(match.group(2))
                modifier = int(match.group(3)) if match.group(3) else 0
                return {"sides": sides, "count": count, "modifier": modifier}
            
            # Default to d20 if no specific dice mentioned
            return {"sides": 20, "count": 1, "modifier": 0}
        
        elif tool_name == "calculate_combat":
            # Extract combat information
            return {
                "attacker_level": 5,  # Default values
                "defender_ac": 15,
                "weapon_damage": "1d8+3",
                "attack_bonus": 0
            }
        
        elif tool_name == "manage_inventory":
            if "list" in user_input.lower():
                return {"action": "list"}
            elif "add" in user_input.lower():
                return {"action": "add", "item_name": "Unknown Item", "quantity": 1}
            else:
                return {"action": "list"}
        
        elif tool_name == "generate_npc":
            npc_types = ["merchant", "guard", "quest_giver", "enemy"]
            for npc_type in npc_types:
                if npc_type in user_input.lower():
                    return {"npc_type": npc_type, "level": 5, "race": "human"}
            return {"npc_type": "merchant", "level": 5, "race": "human"}
        
        return None
    
    def _generate_tool_response(self, user_input: str, tool_results: List[ToolCall], strategy: Dict[str, Any]) -> str:
        """Generate response incorporating tool results"""
        
        # Create response prompt
        tool_results_text = "\n".join([
            f"Tool: {result.tool_name}\nResult: {json.dumps(result.result, indent=2)}"
            for result in tool_results if result.status == ToolStatus.SUCCESS
        ])
        
        response_prompt = f"""
You are a D&D Dungeon Master. Generate a response to the player's input, incorporating the results of tool executions.

Player Input: {user_input}

Tool Results:
{tool_results_text}

Response Focus: {strategy.get('response_focus', 'Provide appropriate response')}

Generate a natural, engaging response that:
1. Addresses the player's input
2. Incorporates tool results naturally
3. Maintains game immersion
4. Follows D&D 5e rules and conventions

Response:
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.primary_model,
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error("Error generating tool response", error=str(e))
            return f"I processed your request using some tools, but encountered an error generating the response: {str(e)}"
    
    def _execute_planning_response(self, strategy: Dict[str, Any], user_input: str) -> str:
        """Execute response using planning system"""
        
        self.state = AgentState.PLANNING
        
        # Create goal based on input
        goal = self._create_goal_from_input(user_input)
        
        # Generate plan
        try:
            plan = generate_plan(goal.id)
            
            # Execute first few tasks
            executed_tasks = []
            for task_id in plan[:3]:  # Execute first 3 tasks
                task = self.planning_system.tasks[task_id]
                outcome = self._execute_task(task)
                self.planning_system.complete_task(task_id, outcome)
                executed_tasks.append(task.name)
            
            response = f"I've created a plan to address your request and executed some initial steps: {', '.join(executed_tasks)}. The plan is now in progress."
            
        except Exception as e:
            self.logger.error("Error in planning response", error=str(e))
            response = f"I attempted to create a plan for your request but encountered an issue: {str(e)}"
        
        self.state = AgentState.IDLE
        return response
    
    def _create_goal_from_input(self, user_input: str) -> Any:
        """Create a goal based on user input"""
        
        # Classify input and create appropriate goal
        input_type = self._classify_input(user_input)
        
        if input_type == "combat_action":
            return create_goal(
                name="Handle Combat Situation",
                description=f"Address the combat situation described: {user_input}",
                goal_type=GoalType.COMBAT,
                priority=8
            )
        elif input_type == "exploration_action":
            return create_goal(
                name="Handle Exploration",
                description=f"Address the exploration request: {user_input}",
                goal_type=GoalType.EXPLORATION,
                priority=6
            )
        elif input_type == "social_action":
            return create_goal(
                name="Handle Social Interaction",
                description=f"Address the social interaction: {user_input}",
                goal_type=GoalType.SOCIAL,
                priority=5
            )
        else:
            return create_goal(
                name="Address Player Request",
                description=f"Address the player's request: {user_input}",
                goal_type=GoalType.ACHIEVEMENT,
                priority=5
            )
    
    def _execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute a planning task"""
        
        # Simplified task execution
        # In a real implementation, this would be more sophisticated
        
        return {
            "status": "completed",
            "summary": f"Executed task: {task.name}",
            "result": "Task completed successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    def _execute_reflection_response(self, strategy: Dict[str, Any], user_input: str) -> str:
        """Execute response using reflection and learning"""
        
        self.state = AgentState.REFLECTING
        
        # Perform reflection
        reflection = self._reflect_on_interaction(user_input)
        
        # Update agent behavior based on reflection
        self._update_behavior_from_reflection(reflection)
        
        response = f"I've reflected on your input and updated my approach. {reflection.get('summary', 'Reflection completed.')}"
        
        self.state = AgentState.IDLE
        return response
    
    def _execute_immediate_response(self, strategy: Dict[str, Any], user_input: str) -> str:
        """Execute immediate response without tools or planning"""
        
        # Generate immediate response using LLM
        response_prompt = f"""
You are a D&D Dungeon Master. Provide an immediate response to the player's input.

Player Input: {user_input}

Current Game State:
- Location: {self.game_state.location}
- Party: {', '.join(f'{name} (HP: {attrs["HP"]})' for name, attrs in self.game_state.party.items())}

Response Focus: {strategy.get('response_focus', 'Provide appropriate response')}

Generate a natural, engaging response that:
1. Addresses the player's input directly
2. Maintains game immersion
3. Follows D&D 5e rules and conventions
4. Is appropriate for the current situation

Response:
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.primary_model,
                messages=[{"role": "user", "content": response_prompt}],
                temperature=0.7,
                max_tokens=600
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error("Error generating immediate response", error=str(e))
            return f"I'm processing your request but encountered an error: {str(e)}"
    
    def _update_performance_metrics(self, start_time: float, strategy: Dict[str, Any]):
        """Update agent performance metrics"""
        
        response_time = time.time() - start_time
        self.performance_metrics["total_interactions"] += 1
        
        # Update average response time
        current_avg = self.performance_metrics["average_response_time"]
        total_interactions = self.performance_metrics["total_interactions"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_interactions - 1) + response_time) / total_interactions
        )
        
        # Update strategy-specific metrics
        strategy_type = strategy.get("strategy", "immediate_response")
        if strategy_type == "planning":
            self.performance_metrics["planning_cycles"] += 1
        elif strategy_type == "reflection":
            self.performance_metrics["reflection_cycles"] += 1
    
    def _store_interaction(self, user_input: str, response: str, strategy: Dict[str, Any]):
        """Store interaction for learning and analysis"""
        
        interaction = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "agent_response": response,
            "strategy": strategy,
            "performance_metrics": self.performance_metrics.copy(),
            "context": self.context.current_situation.copy()
        }
        
        self.interaction_history.append(interaction)
        
        # Store in memory
        create_memory(
            content=f"Interaction: User said '{user_input[:100]}...' and I responded with strategy '{strategy.get('strategy', 'unknown')}'",
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.MEDIUM,
            tags=["interaction", strategy.get("strategy", "unknown")],
            context={"interaction": interaction}
        )
    
    def _reflect_and_learn(self):
        """Perform periodic reflection and learning"""
        
        self.state = AgentState.REFLECTING
        
        # Analyze recent performance
        recent_interactions = self.interaction_history[-10:]
        
        # Calculate performance metrics
        success_rate = len([i for i in recent_interactions if "error" not in i["agent_response"].lower()]) / len(recent_interactions)
        self.performance_metrics["goal_completion_rate"] = success_rate
        
        # Update memory utilization
        memory_stats = self.memory_manager.get_memory_stats()
        self.performance_metrics["memory_utilization"] = memory_stats["short_term_count"] / 100  # Normalized
        
        # Update tool effectiveness
        tool_stats = get_tool_stats()
        if tool_stats["total_executions"] > 0:
            self.performance_metrics["tool_effectiveness"] = tool_stats["successful_executions"] / tool_stats["total_executions"]
        
        # Store reflection
        create_memory(
            content=f"Reflection: Performance analysis shows {success_rate:.1%} success rate over last 10 interactions",
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.MEDIUM,
            tags=["reflection", "performance"],
            context={"performance_metrics": self.performance_metrics.copy()}
        )
        
        self.performance_metrics["reflection_cycles"] += 1
        self.state = AgentState.IDLE
    
    def _reflect_on_interaction(self, user_input: str) -> Dict[str, Any]:
        """Reflect on a specific interaction"""
        
        reflection_prompt = f"""
You are an AI agent reflecting on your interaction with a user. Analyze the interaction and provide insights for improvement.

User Input: {user_input}

Recent Performance:
- Total Interactions: {self.performance_metrics['total_interactions']}
- Average Response Time: {self.performance_metrics['average_response_time']:.2f}s
- Goal Completion Rate: {self.performance_metrics['goal_completion_rate']:.1%}

Provide a reflection in JSON format:
{{
    "summary": "Brief summary of reflection",
    "strengths": ["list", "of", "strengths"],
    "weaknesses": ["list", "of", "weaknesses"],
    "improvements": ["list", "of", "improvements"],
    "learning_points": ["key", "learning", "points"]
}}
"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm.primary_model,
                messages=[{"role": "user", "content": reflection_prompt}],
                temperature=0.3,
                max_tokens=400
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error("Error in reflection", error=str(e))
            return {
                "summary": "Reflection failed due to error",
                "strengths": [],
                "weaknesses": ["Reflection system error"],
                "improvements": ["Fix reflection system"],
                "learning_points": []
            }
    
    def _update_behavior_from_reflection(self, reflection: Dict[str, Any]):
        """Update agent behavior based on reflection"""
        
        # Store reflection insights
        create_memory(
            content=f"Reflection insights: {reflection.get('summary', 'No summary')}",
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.HIGH,
            tags=["reflection", "learning"],
            context={"reflection": reflection}
        )
        
        # Apply improvements (simplified)
        improvements = reflection.get("improvements", [])
        for improvement in improvements:
            # In a real implementation, this would update agent behavior
            self.logger.info(f"Applying improvement: {improvement}")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        
        stats = {
            "agent_state": self.state.value,
            "agent_mode": self.mode.value,
            "performance_metrics": self.performance_metrics.copy(),
            "subsystem_stats": {
                "memory": self.memory_manager.get_memory_stats(),
                "planning": self.planning_system.get_planning_stats(),
                "tools": get_tool_stats()
            },
            "context": {
                "active_goals": len(self.context.active_goals),
                "recent_actions": len(self.context.recent_actions),
                "environmental_state": self.context.environmental_state
            },
            "interaction_history_length": len(self.interaction_history)
        }
        
        return stats
    
    def display_agent_status(self):
        """Display agent status in a nice format"""
        
        stats = self.get_agent_stats()
        
        # Create status panel
        status_text = f"""
🤖 Agent Status: {stats['agent_state'].upper()}
🎯 Mode: {stats['agent_mode'].upper()}
📊 Total Interactions: {stats['performance_metrics']['total_interactions']}
⚡ Avg Response Time: {stats['performance_metrics']['average_response_time']:.2f}s
🎯 Success Rate: {stats['performance_metrics']['goal_completion_rate']:.1%}
🧠 Memory Utilization: {stats['performance_metrics']['memory_utilization']:.1%}
🔧 Tool Effectiveness: {stats['performance_metrics']['tool_effectiveness']:.1%}
        """.strip()
        
        console.print(Panel(status_text, title="Agentic Agent Status", border_style="blue"))
        
        # Display subsystem stats
        self._display_subsystem_stats(stats["subsystem_stats"])
    
    def _display_subsystem_stats(self, subsystem_stats: Dict[str, Any]):
        """Display subsystem statistics"""
        
        table = Table(title="Subsystem Statistics")
        table.add_column("Subsystem", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Value", style="yellow")
        
        # Memory stats
        memory_stats = subsystem_stats["memory"]
        table.add_row("Memory", "Short-term", str(memory_stats["short_term_count"]))
        table.add_row("Memory", "Hit Rate", f"{memory_stats['access_stats']['memory_hit_rate']:.1%}")
        
        # Planning stats
        planning_stats = subsystem_stats["planning"]
        table.add_row("Planning", "Active Goals", str(planning_stats["active_goals"]))
        table.add_row("Planning", "Pending Tasks", str(planning_stats["pending_tasks"]))
        
        # Tools stats
        tools_stats = subsystem_stats["tools"]
        table.add_row("Tools", "Available", str(tools_stats["available_tools"]))
        table.add_row("Tools", "Success Rate", f"{(tools_stats['successful_executions'] / max(tools_stats['total_executions'], 1) * 100):.1f}%")
        
        console.print(table)


# Global agentic agent instance
agentic_agent = AgenticAgent()


# Utility functions for easy access
def process_input(user_input: str) -> str:
    """Process user input through the agentic agent"""
    return agentic_agent.process_input(user_input)


def get_agent_stats() -> Dict[str, Any]:
    """Get agent statistics"""
    return agentic_agent.get_agent_stats()


def display_agent_status():
    """Display agent status"""
    agentic_agent.display_agent_status() 