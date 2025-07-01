"""
Tools System for Agentic AI
===========================

This module implements a sophisticated tools system that demonstrates key AI/ML concepts:

1. **ReAct Pattern**: Reasoning and Acting in an interleaved manner
2. **Tool Integration**: Seamless integration of external tools with LLM reasoning
3. **Tool Validation**: Ensuring tool inputs are valid and safe
4. **Tool Selection**: Choosing the right tool for the job

Key AI/ML Concepts Demonstrated:
- Tool Calling and Execution
- Input Validation and Sanitization
- Error Handling and Fallbacks
- Tool Composition and Chaining
"""

import json
import random
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
from rich.console import Console
from rich.table import Table

from config import get_config
from memory_system import create_memory, MemoryType, MemoryPriority

logger = structlog.get_logger(__name__)
console = Console()


class ToolType(Enum):
    """Types of tools - demonstrates different capabilities"""
    DICE_ROLLING = "dice_rolling"
    COMBAT_CALCULATOR = "combat_calculator"
    INVENTORY_MANAGER = "inventory_manager"
    NPC_GENERATOR = "npc_generator"
    LOCATION_GENERATOR = "location_generator"
    QUEST_GENERATOR = "quest_generator"
    MATH_CALCULATOR = "math_calculator"
    TEXT_PROCESSOR = "text_processor"


class ToolStatus(Enum):
    """Tool execution status"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    TOOL_UNAVAILABLE = "tool_unavailable"


@dataclass
class ToolCall:
    """
    Tool Call Data Structure - Demonstrates Tool Execution Tracking
    
    Key Concepts:
    - Tool Identification: Which tool to call
    - Input Parameters: What data to pass to the tool
    - Execution Tracking: When and how the tool was called
    - Result Storage: What the tool returned
    """
    
    id: str
    tool_name: str
    tool_type: ToolType
    parameters: Dict[str, Any]
    timestamp: datetime
    status: ToolStatus
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool call to dictionary"""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type.value,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "context": self.context
        }


@dataclass
class Tool:
    """
    Tool Data Structure - Demonstrates Tool Definition
    
    Key Concepts:
    - Tool Specification: What the tool does and how to use it
    - Input Schema: What parameters the tool accepts
    - Output Schema: What the tool returns
    - Validation Rules: How to validate inputs
    """
    
    name: str
    tool_type: ToolType
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    function: Callable
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    timeout: int = 30
    enabled: bool = True
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    def validate_input(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate tool input parameters
        
        Key Concepts:
        - Schema Validation: Ensuring inputs match expected format
        - Business Logic Validation: Checking domain-specific rules
        - Security Validation: Preventing malicious inputs
        """
        
        # Check required parameters
        for param_name, param_info in self.input_schema.items():
            if param_info.get("required", False) and param_name not in parameters:
                return False, f"Missing required parameter: {param_name}"
            
            if param_name in parameters:
                param_value = parameters[param_name]
                param_type = param_info.get("type")
                
                # Type validation
                if param_type == "integer":
                    if not isinstance(param_value, int):
                        return False, f"Parameter {param_name} must be an integer"
                elif param_type == "string":
                    if not isinstance(param_value, str):
                        return False, f"Parameter {param_name} must be a string"
                elif param_type == "list":
                    if not isinstance(param_value, list):
                        return False, f"Parameter {param_name} must be a list"
                
                # Range validation
                if "min" in param_info and param_value < param_info["min"]:
                    return False, f"Parameter {param_name} must be >= {param_info['min']}"
                if "max" in param_info and param_value > param_info["max"]:
                    return False, f"Parameter {param_name} must be <= {param_info['max']}"
        
        # Custom validation rules
        for rule in self.validation_rules:
            if not self._evaluate_validation_rule(rule, parameters):
                return False, rule.get("error_message", "Validation failed")
        
        return True, None
    
    def _evaluate_validation_rule(self, rule: Dict[str, Any], parameters: Dict[str, Any]) -> bool:
        """Evaluate a custom validation rule"""
        rule_type = rule.get("type")
        
        if rule_type == "custom":
            # Custom validation function
            validation_func = rule.get("function")
            if validation_func:
                return validation_func(parameters)
        
        elif rule_type == "dependency":
            # Parameter dependency validation
            param1 = rule.get("param1")
            param2 = rule.get("param2")
            condition = rule.get("condition")
            
            if param1 in parameters and param2 in parameters:
                val1 = parameters[param1]
                val2 = parameters[param2]
                
                if condition == "greater_than":
                    return val1 > val2
                elif condition == "less_than":
                    return val1 < val2
                elif condition == "equals":
                    return val1 == val2
        
        return True
    
    def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolCall:
        """
        Execute the tool with given parameters
        
        Key Concepts:
        - Input Validation: Ensuring parameters are valid
        - Execution Tracking: Monitoring tool performance
        - Error Handling: Graceful handling of failures
        - Result Processing: Formatting and validating outputs
        """
        
        start_time = time.time()
        tool_call = ToolCall(
            id=str(uuid.uuid4()),
            tool_name=self.name,
            tool_type=self.tool_type,
            parameters=parameters,
            timestamp=datetime.now(),
            status=ToolStatus.FAILED,  # Default to failed, update on success
            context=context or {}
        )
        
        try:
            # Validate input
            is_valid, error_message = self.validate_input(parameters)
            if not is_valid:
                tool_call.error_message = error_message
                tool_call.status = ToolStatus.INVALID_INPUT
                return tool_call
            
            # Execute tool function
            result = self.function(**parameters)
            
            # Validate output
            if not self._validate_output(result):
                tool_call.error_message = "Invalid output from tool"
                tool_call.status = ToolStatus.FAILED
                return tool_call
            
            # Update tool call
            tool_call.result = result
            tool_call.status = ToolStatus.SUCCESS
            tool_call.execution_time = time.time() - start_time
            
            # Update tool statistics
            self.usage_count += 1
            self.last_used = datetime.now()
            
            logger.info(
                "Tool executed successfully",
                tool_name=self.name,
                execution_time=tool_call.execution_time,
                parameters=parameters
            )
            
        except Exception as e:
            tool_call.error_message = str(e)
            tool_call.status = ToolStatus.FAILED
            tool_call.execution_time = time.time() - start_time
            
            logger.error(
                "Tool execution failed",
                tool_name=self.name,
                error=str(e),
                parameters=parameters
            )
        
        return tool_call
    
    def _validate_output(self, result: Any) -> bool:
        """Validate tool output"""
        # Basic output validation
        if result is None:
            return False
        
        # Check output schema if defined
        if self.output_schema:
            expected_type = self.output_schema.get("type")
            if expected_type == "integer" and not isinstance(result, int):
                return False
            elif expected_type == "string" and not isinstance(result, str):
                return False
            elif expected_type == "dict" and not isinstance(result, dict):
                return False
        
        return True


class ToolsSystem:
    """
    Tools System - Orchestrates Tool Management and Execution
    
    This class demonstrates several key AI/ML concepts:
    
    1. **Tool Registry**: Managing available tools
    2. **Tool Selection**: Choosing appropriate tools for tasks
    3. **Execution Orchestration**: Coordinating tool execution
    4. **Result Processing**: Handling and formatting tool outputs
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(component="tools_system")
        
        # Tool registry
        self.tools: Dict[str, Tool] = {}
        
        # Execution history
        self.execution_history: List[ToolCall] = []
        
        # Tool statistics
        self.tool_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "most_used_tool": None,
            "tool_usage_counts": {}
        }
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all available tools"""
        
        # Dice Rolling Tool
        if self.config.tools.enable_dice_rolling:
            self.register_tool(Tool(
                name="roll_dice",
                tool_type=ToolType.DICE_ROLLING,
                description="Roll dice for D&D gameplay",
                input_schema={
                    "sides": {"type": "integer", "required": True, "min": 2, "max": 100},
                    "count": {"type": "integer", "required": False, "min": 1, "max": 10, "default": 1},
                    "modifier": {"type": "integer", "required": False, "default": 0}
                },
                output_schema={"type": "dict"},
                function=self._roll_dice,
                validation_rules=[
                    {
                        "type": "custom",
                        "function": lambda params: params.get("sides", 0) > 1,
                        "error_message": "Dice must have at least 2 sides"
                    }
                ]
            ))
        
        # Combat Calculator Tool
        if self.config.tools.enable_combat_calculator:
            self.register_tool(Tool(
                name="calculate_combat",
                tool_type=ToolType.COMBAT_CALCULATOR,
                description="Calculate combat outcomes",
                input_schema={
                    "attacker_level": {"type": "integer", "required": True, "min": 1, "max": 20},
                    "defender_ac": {"type": "integer", "required": True, "min": 1, "max": 30},
                    "weapon_damage": {"type": "string", "required": True},
                    "attack_bonus": {"type": "integer", "required": False, "default": 0}
                },
                output_schema={"type": "dict"},
                function=self._calculate_combat
            ))
        
        # Inventory Manager Tool
        if self.config.tools.enable_inventory_manager:
            self.register_tool(Tool(
                name="manage_inventory",
                tool_type=ToolType.INVENTORY_MANAGER,
                description="Manage player inventory",
                input_schema={
                    "action": {"type": "string", "required": True, "enum": ["add", "remove", "list", "search"]},
                    "item_name": {"type": "string", "required": False},
                    "quantity": {"type": "integer", "required": False, "min": 1, "default": 1}
                },
                output_schema={"type": "dict"},
                function=self._manage_inventory
            ))
        
        # NPC Generator Tool
        if self.config.tools.enable_npc_generator:
            self.register_tool(Tool(
                name="generate_npc",
                tool_type=ToolType.NPC_GENERATOR,
                description="Generate NPCs for the game",
                input_schema={
                    "npc_type": {"type": "string", "required": True, "enum": ["merchant", "guard", "quest_giver", "enemy"]},
                    "level": {"type": "integer", "required": False, "min": 1, "max": 20, "default": 5},
                    "race": {"type": "string", "required": False, "enum": ["human", "elf", "dwarf", "orc", "random"]}
                },
                output_schema={"type": "dict"},
                function=self._generate_npc
            ))
        
        self.logger.info(f"Initialized {len(self.tools)} tools")
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        self.tool_stats["tool_usage_counts"][tool.name] = 0
        
        self.logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools"""
        return [
            {
                "name": tool.name,
                "type": tool.tool_type.value,
                "description": tool.description,
                "enabled": tool.enabled,
                "usage_count": tool.usage_count
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolCall:
        """
        Execute a tool - Demonstrates Tool Execution
        
        Key Concepts:
        - Tool Lookup: Finding the right tool
        - Parameter Validation: Ensuring inputs are valid
        - Execution Monitoring: Tracking performance
        - Result Processing: Handling outputs
        """
        
        if tool_name not in self.tools:
            tool_call = ToolCall(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                tool_type=ToolType.DICE_ROLLING,  # Default type
                parameters=parameters,
                timestamp=datetime.now(),
                status=ToolStatus.TOOL_UNAVAILABLE,
                error_message=f"Tool '{tool_name}' not found"
            )
            return tool_call
        
        tool = self.tools[tool_name]
        
        if not tool.enabled:
            tool_call = ToolCall(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                tool_type=tool.tool_type,
                parameters=parameters,
                timestamp=datetime.now(),
                status=ToolStatus.TOOL_UNAVAILABLE,
                error_message=f"Tool '{tool_name}' is disabled"
            )
            return tool_call
        
        # Execute tool
        tool_call = tool.execute(parameters, context)
        
        # Update statistics
        self._update_tool_stats(tool_call)
        
        # Store in history
        self.execution_history.append(tool_call)
        
        # Store in memory if successful
        if tool_call.status == ToolStatus.SUCCESS:
            create_memory(
                content=f"Tool executed: {tool_name} with result: {str(tool_call.result)[:100]}",
                memory_type=MemoryType.PROCEDURAL,
                priority=MemoryPriority.MEDIUM,
                tags=["tool_execution", tool.tool_type.value],
                context={"tool_call": tool_call.to_dict()}
            )
        
        return tool_call
    
    def _update_tool_stats(self, tool_call: ToolCall):
        """Update tool execution statistics"""
        self.tool_stats["total_executions"] += 1
        
        if tool_call.status == ToolStatus.SUCCESS:
            self.tool_stats["successful_executions"] += 1
        else:
            self.tool_stats["failed_executions"] += 1
        
        if tool_call.execution_time:
            # Update average execution time
            current_avg = self.tool_stats["average_execution_time"]
            total_executions = self.tool_stats["total_executions"]
            self.tool_stats["average_execution_time"] = (
                (current_avg * (total_executions - 1) + tool_call.execution_time) / total_executions
            )
        
        # Update usage counts
        tool_name = tool_call.tool_name
        self.tool_stats["tool_usage_counts"][tool_name] = self.tool_stats["tool_usage_counts"].get(tool_name, 0) + 1
        
        # Update most used tool
        most_used = max(self.tool_stats["tool_usage_counts"].items(), key=lambda x: x[1])
        self.tool_stats["most_used_tool"] = most_used[0]
    
    def select_tool(self, task_description: str, available_tools: Optional[List[str]] = None) -> Optional[str]:
        """
        Select the most appropriate tool for a task
        
        Key Concepts:
        - Tool Selection: Choosing the right tool for the job
        - Semantic Matching: Understanding task requirements
        - Tool Capabilities: Matching tools to tasks
        """
        
        if available_tools is None:
            available_tools = list(self.tools.keys())
        
        # Simple keyword-based tool selection
        # In a real implementation, this would use an LLM for semantic matching
        
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["roll", "dice", "random"]):
            if "roll_dice" in available_tools:
                return "roll_dice"
        
        if any(word in task_lower for word in ["combat", "fight", "attack", "damage"]):
            if "calculate_combat" in available_tools:
                return "calculate_combat"
        
        if any(word in task_lower for word in ["inventory", "item", "equipment"]):
            if "manage_inventory" in available_tools:
                return "manage_inventory"
        
        if any(word in task_lower for word in ["npc", "character", "person"]):
            if "generate_npc" in available_tools:
                return "generate_npc"
        
        return None
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get comprehensive tool statistics"""
        stats = self.tool_stats.copy()
        
        # Add recent execution history
        recent_executions = self.execution_history[-10:]  # Last 10 executions
        stats["recent_executions"] = [execution.to_dict() for execution in recent_executions]
        
        # Add tool availability
        stats["available_tools"] = len([tool for tool in self.tools.values() if tool.enabled])
        stats["total_tools"] = len(self.tools)
        
        return stats
    
    # Tool Implementation Functions
    
    def _roll_dice(self, sides: int, count: int = 1, modifier: int = 0) -> Dict[str, Any]:
        """Roll dice for D&D gameplay"""
        results = []
        total = 0
        
        for _ in range(count):
            roll = random.randint(1, sides)
            results.append(roll)
            total += roll
        
        total += modifier
        
        return {
            "dice": f"{count}d{sides}",
            "modifier": modifier,
            "individual_rolls": results,
            "total": total,
            "natural_rolls": results.copy()
        }
    
    def _calculate_combat(self, attacker_level: int, defender_ac: int, weapon_damage: str, attack_bonus: int = 0) -> Dict[str, Any]:
        """Calculate combat outcomes"""
        # Parse weapon damage (e.g., "1d8+3")
        damage_parts = weapon_damage.split("+")
        dice_part = damage_parts[0]
        bonus = int(damage_parts[1]) if len(damage_parts) > 1 else 0
        
        # Parse dice (e.g., "1d8")
        dice_count, dice_sides = map(int, dice_part.split("d"))
        
        # Calculate attack roll
        attack_roll = random.randint(1, 20) + attack_bonus + (attacker_level // 4)  # Proficiency bonus
        
        # Determine hit
        hit = attack_roll >= defender_ac
        critical_hit = attack_roll == 20
        
        # Calculate damage
        if hit:
            if critical_hit:
                # Critical hit: double the dice
                damage = sum(random.randint(1, dice_sides) for _ in range(dice_count * 2)) + bonus
            else:
                damage = sum(random.randint(1, dice_sides) for _ in range(dice_count)) + bonus
        else:
            damage = 0
        
        return {
            "attack_roll": attack_roll,
            "defender_ac": defender_ac,
            "hit": hit,
            "critical_hit": critical_hit,
            "damage": damage,
            "weapon_damage": weapon_damage
        }
    
    def _manage_inventory(self, action: str, item_name: Optional[str] = None, quantity: int = 1) -> Dict[str, Any]:
        """Manage player inventory"""
        # This is a simplified inventory system
        # In a real implementation, this would interact with the game state
        
        if action == "list":
            return {
                "action": "list",
                "inventory": [
                    {"name": "Sword", "quantity": 1, "type": "weapon"},
                    {"name": "Health Potion", "quantity": 3, "type": "consumable"},
                    {"name": "Gold", "quantity": 150, "type": "currency"}
                ]
            }
        elif action == "add":
            return {
                "action": "add",
                "item": item_name,
                "quantity": quantity,
                "status": "added"
            }
        elif action == "remove":
            return {
                "action": "remove",
                "item": item_name,
                "quantity": quantity,
                "status": "removed"
            }
        elif action == "search":
            return {
                "action": "search",
                "item": item_name,
                "found": item_name in ["Sword", "Health Potion", "Gold"]
            }
        
        return {"error": "Invalid action"}
    
    def _generate_npc(self, npc_type: str, level: int = 5, race: str = "human") -> Dict[str, Any]:
        """Generate NPCs for the game"""
        npc_templates = {
            "merchant": {
                "name": "Merchant",
                "personality": "Friendly and business-minded",
                "inventory": ["various goods", "gold"],
                "dialogue_style": "Professional and helpful"
            },
            "guard": {
                "name": "Guard",
                "personality": "Vigilant and protective",
                "inventory": ["sword", "shield", "armor"],
                "dialogue_style": "Direct and authoritative"
            },
            "quest_giver": {
                "name": "Quest Giver",
                "personality": "Concerned and hopeful",
                "inventory": ["quest items", "rewards"],
                "dialogue_style": "Urgent and pleading"
            },
            "enemy": {
                "name": "Enemy",
                "personality": "Hostile and aggressive",
                "inventory": ["weapons", "loot"],
                "dialogue_style": "Threatening and confrontational"
            }
        }
        
        template = npc_templates.get(npc_type, npc_templates["merchant"])
        
        # Generate random name
        names = {
            "human": ["John", "Sarah", "Michael", "Emma", "David"],
            "elf": ["Elrond", "Galadriel", "Legolas", "Arwen", "Thranduil"],
            "dwarf": ["Gimli", "Thorin", "Balin", "Dwalin", "Bifur"],
            "orc": ["Grom", "Karg", "Mogka", "Ragash", "Zug"]
        }
        
        if race == "random":
            race = random.choice(list(names.keys()))
        
        name = random.choice(names.get(race, names["human"]))
        
        return {
            "name": f"{name} the {template['name']}",
            "type": npc_type,
            "race": race,
            "level": level,
            "personality": template["personality"],
            "inventory": template["inventory"],
            "dialogue_style": template["dialogue_style"],
            "stats": {
                "strength": random.randint(8, 18),
                "dexterity": random.randint(8, 18),
                "constitution": random.randint(8, 18),
                "intelligence": random.randint(8, 18),
                "wisdom": random.randint(8, 18),
                "charisma": random.randint(8, 18)
            }
        }
    
    def display_tool_stats(self):
        """Display tool statistics in a nice format"""
        stats = self.get_tool_stats()
        
        table = Table(title="Tool Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Executions", str(stats["total_executions"]))
        table.add_row("Successful Executions", str(stats["successful_executions"]))
        table.add_row("Failed Executions", str(stats["failed_executions"]))
        table.add_row("Success Rate", f"{(stats['successful_executions'] / max(stats['total_executions'], 1) * 100):.1f}%")
        table.add_row("Average Execution Time", f"{stats['average_execution_time']:.3f}s")
        table.add_row("Most Used Tool", stats["most_used_tool"] or "None")
        table.add_row("Available Tools", str(stats["available_tools"]))
        
        console.print(table)


# Global tools system instance
tools_system = ToolsSystem()


# Utility functions for easy access
def execute_tool(*args, **kwargs) -> ToolCall:
    """Execute a tool"""
    return tools_system.execute_tool(*args, **kwargs)


def select_tool(*args, **kwargs) -> Optional[str]:
    """Select appropriate tool for task"""
    return tools_system.select_tool(*args, **kwargs)


def get_tool_stats() -> Dict[str, Any]:
    """Get tool statistics"""
    return tools_system.get_tool_stats()


def list_tools() -> List[Dict[str, Any]]:
    """List available tools"""
    return tools_system.list_tools() 