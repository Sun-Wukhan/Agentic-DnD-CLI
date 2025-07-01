"""
Planning System for Agentic AI
==============================

This module implements a sophisticated planning system that demonstrates key AI/ML concepts:

1. **Goal-Directed Behavior**: Breaking down high-level goals into actionable tasks
2. **Task Decomposition**: Hierarchical planning with subtasks
3. **Replanning Mechanisms**: Adapting plans based on changing circumstances
4. **Priority Management**: Balancing multiple goals and constraints

Key AI/ML Concepts Demonstrated:
- Hierarchical Task Networks (HTN)
- Goal-Oriented Planning
- Dynamic Replanning
- Constraint Satisfaction
"""

import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog
from rich.console import Console
from rich.table import Table

from config import get_config
from memory_system import create_memory, MemoryType, MemoryPriority

logger = structlog.get_logger(__name__)
console = Console()


class GoalStatus(Enum):
    """Goal status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class GoalType(Enum):
    """Types of goals - demonstrates different planning strategies"""
    SURVIVAL = "survival"          # Immediate survival needs
    EXPLORATION = "exploration"    # Discovering new areas
    COMBAT = "combat"              # Engaging in combat
    SOCIAL = "social"              # Interacting with NPCs
    ACHIEVEMENT = "achievement"    # Long-term accomplishments
    MAINTENANCE = "maintenance"    # Maintaining resources/health


@dataclass
class Goal:
    """
    Goal Data Structure - Demonstrates Goal Representation
    
    Key Concepts:
    - Goal Hierarchy: Goals can have sub-goals
    - Priority System: Goals have different importance levels
    - Success Criteria: Clear conditions for goal completion
    - Temporal Constraints: Time limits and deadlines
    """
    
    id: str
    name: str
    description: str
    goal_type: GoalType
    priority: int  # 1-10, higher is more important
    status: GoalStatus
    created_at: datetime
    deadline: Optional[datetime]
    success_criteria: List[str]
    failure_criteria: List[str]
    sub_goals: List[str] = field(default_factory=list)  # IDs of sub-goals
    parent_goal: Optional[str] = None  # ID of parent goal
    progress: float = 0.0  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def is_completed(self) -> bool:
        """Check if goal is completed"""
        return self.status == GoalStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if goal has failed"""
        return self.status in [GoalStatus.FAILED, GoalStatus.ABANDONED]
    
    def is_active(self) -> bool:
        """Check if goal is still active"""
        return self.status in [GoalStatus.PENDING, GoalStatus.IN_PROGRESS]
    
    def update_progress(self, progress: float):
        """Update goal progress"""
        self.progress = max(0.0, min(1.0, progress))
        if self.progress >= 1.0:
            self.status = GoalStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal_type": self.goal_type.value,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "success_criteria": self.success_criteria,
            "failure_criteria": self.failure_criteria,
            "sub_goals": self.sub_goals,
            "parent_goal": self.parent_goal,
            "progress": self.progress,
            "confidence": self.confidence,
            "context": self.context
        }


@dataclass
class Task:
    """
    Task Data Structure - Demonstrates Task Representation
    
    Key Concepts:
    - Task Dependencies: Tasks can depend on other tasks
    - Resource Requirements: What resources are needed
    - Execution Conditions: When tasks can be executed
    - Outcome Tracking: What happened when task was executed
    """
    
    id: str
    name: str
    description: str
    goal_id: str
    status: TaskStatus
    priority: int  # 1-10, higher is more important
    created_at: datetime
    estimated_duration: Optional[int] = None  # seconds
    dependencies: List[str] = field(default_factory=list)  # IDs of dependent tasks
    required_resources: Dict[str, Any] = field(default_factory=dict)
    execution_conditions: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)  # What actions to take
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready to execute (dependencies satisfied)"""
        return all(dep in completed_tasks for dep in self.dependencies)
    
    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.status == TaskStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if task has failed"""
        return self.status == TaskStatus.FAILED
    
    def add_outcome(self, outcome: Dict[str, Any]):
        """Add execution outcome"""
        outcome["timestamp"] = datetime.now().isoformat()
        self.outcomes.append(outcome)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal_id": self.goal_id,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "estimated_duration": self.estimated_duration,
            "dependencies": self.dependencies,
            "required_resources": self.required_resources,
            "execution_conditions": self.execution_conditions,
            "actions": self.actions,
            "outcomes": self.outcomes,
            "context": self.context
        }


class PlanningSystem:
    """
    Planning System - Orchestrates Goal and Task Management
    
    This class demonstrates several key AI/ML concepts:
    
    1. **Hierarchical Planning**: Breaking complex goals into manageable tasks
    2. **Dynamic Replanning**: Adapting plans based on new information
    3. **Constraint Satisfaction**: Ensuring plans are feasible
    4. **Priority Management**: Balancing multiple competing goals
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(component="planning_system")
        
        # Goal and task storage
        self.goals: Dict[str, Goal] = {}
        self.tasks: Dict[str, Task] = {}
        
        # Planning state
        self.current_plan: List[str] = []  # Task IDs in execution order
        self.execution_history: List[Dict[str, Any]] = []
        
        # Planning statistics
        self.planning_stats = {
            "total_goals_created": 0,
            "total_tasks_created": 0,
            "successful_plans": 0,
            "failed_plans": 0,
            "replanning_events": 0,
            "average_planning_time": 0.0
        }
    
    def create_goal(
        self,
        name: str,
        description: str,
        goal_type: GoalType,
        priority: int = 5,
        deadline: Optional[datetime] = None,
        success_criteria: Optional[List[str]] = None,
        failure_criteria: Optional[List[str]] = None,
        parent_goal: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Goal:
        """
        Create a new goal - Demonstrates Goal Formation
        
        Key Concepts:
        - Goal Specification: Clear definition of what needs to be achieved
        - Priority Assignment: Determining relative importance
        - Success Criteria: Measurable conditions for completion
        - Hierarchical Structure: Goals can have parent-child relationships
        """
        
        goal = Goal(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            goal_type=goal_type,
            priority=priority,
            status=GoalStatus.PENDING,
            created_at=datetime.now(),
            deadline=deadline,
            success_criteria=success_criteria or [],
            failure_criteria=failure_criteria or [],
            parent_goal=parent_goal,
            context=context or {}
        )
        
        self.goals[goal.id] = goal
        self.planning_stats["total_goals_created"] += 1
        
        # Store goal in memory
        create_memory(
            content=f"Goal created: {name} - {description}",
            memory_type=MemoryType.SEMANTIC,
            priority=MemoryPriority.HIGH if priority >= 7 else MemoryPriority.MEDIUM,
            tags=["goal", goal_type.value],
            context={"goal_id": goal.id, "priority": priority}
        )
        
        self.logger.info(
            "Goal created",
            goal_id=goal.id,
            goal_name=name,
            goal_type=goal_type.value,
            priority=priority
        )
        
        return goal
    
    def create_task(
        self,
        name: str,
        description: str,
        goal_id: str,
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        required_resources: Optional[Dict[str, Any]] = None,
        execution_conditions: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        estimated_duration: Optional[int] = None
    ) -> Task:
        """
        Create a new task - Demonstrates Task Decomposition
        
        Key Concepts:
        - Task Specification: Clear definition of what needs to be done
        - Dependency Management: Tasks can depend on other tasks
        - Resource Requirements: What resources are needed
        - Action Specification: Concrete steps to take
        """
        
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            goal_id=goal_id,
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=datetime.now(),
            estimated_duration=estimated_duration,
            dependencies=dependencies or [],
            required_resources=required_resources or {},
            execution_conditions=execution_conditions or [],
            actions=actions or []
        )
        
        self.tasks[task.id] = task
        self.planning_stats["total_tasks_created"] += 1
        
        # Add task to goal's task list
        goal = self.goals[goal_id]
        if not hasattr(goal, 'tasks'):
            goal.tasks = []
        goal.tasks.append(task.id)
        
        self.logger.info(
            "Task created",
            task_id=task.id,
            task_name=name,
            goal_id=goal_id,
            priority=priority
        )
        
        return task
    
    def decompose_goal(self, goal_id: str) -> List[Task]:
        """
        Decompose a goal into tasks - Demonstrates Hierarchical Planning
        
        Key Concepts:
        - Goal Decomposition: Breaking complex goals into simpler tasks
        - Task Generation: Creating specific, actionable tasks
        - Dependency Analysis: Understanding task relationships
        - Resource Planning: Identifying what resources are needed
        """
        
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.goals[goal_id]
        tasks = []
        
        # Use LLM to decompose goal (simplified version)
        # In a real implementation, this would use an LLM to generate tasks
        if goal.goal_type == GoalType.EXPLORATION:
            tasks.extend(self._decompose_exploration_goal(goal))
        elif goal.goal_type == GoalType.COMBAT:
            tasks.extend(self._decompose_combat_goal(goal))
        elif goal.goal_type == GoalType.SOCIAL:
            tasks.extend(self._decompose_social_goal(goal))
        else:
            tasks.extend(self._decompose_generic_goal(goal))
        
        # Create tasks
        created_tasks = []
        for task_spec in tasks:
            task = self.create_task(
                name=task_spec["name"],
                description=task_spec["description"],
                goal_id=goal_id,
                priority=task_spec.get("priority", goal.priority),
                dependencies=task_spec.get("dependencies", []),
                required_resources=task_spec.get("required_resources", {}),
                execution_conditions=task_spec.get("execution_conditions", []),
                actions=task_spec.get("actions", [])
            )
            created_tasks.append(task)
        
        self.logger.info(
            "Goal decomposed",
            goal_id=goal_id,
            tasks_created=len(created_tasks)
        )
        
        return created_tasks
    
    def _decompose_exploration_goal(self, goal: Goal) -> List[Dict[str, Any]]:
        """Decompose exploration goal into tasks"""
        return [
            {
                "name": "Assess Current Location",
                "description": "Analyze the current environment and identify potential paths",
                "priority": goal.priority,
                "actions": ["look around", "check for exits", "identify landmarks"]
            },
            {
                "name": "Choose Exploration Direction",
                "description": "Select the most promising direction to explore",
                "priority": goal.priority,
                "dependencies": ["Assess Current Location"],
                "actions": ["evaluate options", "select best path"]
            },
            {
                "name": "Navigate to New Area",
                "description": "Move to the selected area while avoiding dangers",
                "priority": goal.priority,
                "dependencies": ["Choose Exploration Direction"],
                "actions": ["move carefully", "check for traps", "maintain awareness"]
            }
        ]
    
    def _decompose_combat_goal(self, goal: Goal) -> List[Dict[str, Any]]:
        """Decompose combat goal into tasks"""
        return [
            {
                "name": "Assess Threat",
                "description": "Evaluate the enemy's capabilities and weaknesses",
                "priority": goal.priority,
                "actions": ["observe enemy", "identify weaknesses", "assess danger level"]
            },
            {
                "name": "Prepare for Combat",
                "description": "Get ready for battle with appropriate equipment and positioning",
                "priority": goal.priority,
                "dependencies": ["Assess Threat"],
                "actions": ["equip weapons", "position strategically", "cast buffs"]
            },
            {
                "name": "Execute Combat Strategy",
                "description": "Fight the enemy using the best available tactics",
                "priority": goal.priority,
                "dependencies": ["Prepare for Combat"],
                "actions": ["attack enemy", "use special abilities", "adapt tactics"]
            }
        ]
    
    def _decompose_social_goal(self, goal: Goal) -> List[Dict[str, Any]]:
        """Decompose social goal into tasks"""
        return [
            {
                "name": "Identify NPC",
                "description": "Find and approach the target NPC",
                "priority": goal.priority,
                "actions": ["look around", "approach carefully", "greet appropriately"]
            },
            {
                "name": "Establish Communication",
                "description": "Initiate conversation and establish rapport",
                "priority": goal.priority,
                "dependencies": ["Identify NPC"],
                "actions": ["start conversation", "show interest", "build rapport"]
            },
            {
                "name": "Achieve Social Objective",
                "description": "Accomplish the specific social goal (trade, quest, etc.)",
                "priority": goal.priority,
                "dependencies": ["Establish Communication"],
                "actions": ["negotiate", "complete objective", "maintain relationship"]
            }
        ]
    
    def _decompose_generic_goal(self, goal: Goal) -> List[Dict[str, Any]]:
        """Decompose generic goal into tasks"""
        return [
            {
                "name": f"Plan {goal.name}",
                "description": f"Create a plan to achieve: {goal.description}",
                "priority": goal.priority,
                "actions": ["analyze situation", "identify steps", "create plan"]
            },
            {
                "name": f"Execute {goal.name}",
                "description": f"Carry out the plan for: {goal.description}",
                "priority": goal.priority,
                "dependencies": [f"Plan {goal.name}"],
                "actions": ["follow plan", "adapt as needed", "complete objective"]
            }
        ]
    
    def generate_plan(self, goal_id: str) -> List[str]:
        """
        Generate execution plan for a goal - Demonstrates Planning Algorithms
        
        Key Concepts:
        - Task Scheduling: Determining optimal execution order
        - Dependency Resolution: Ensuring dependencies are satisfied
        - Resource Allocation: Managing resource constraints
        - Priority Optimization: Balancing multiple objectives
        """
        
        start_time = time.time()
        
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.goals[goal_id]
        
        # Get all tasks for this goal
        goal_tasks = [task for task in self.tasks.values() if task.goal_id == goal_id]
        
        if not goal_tasks:
            # Decompose goal if no tasks exist
            goal_tasks = self.decompose_goal(goal_id)
        
        # Create dependency graph
        dependency_graph = self._build_dependency_graph(goal_tasks)
        
        # Generate execution order using topological sort
        execution_order = self._topological_sort(dependency_graph)
        
        # Validate plan
        if not self._validate_plan(execution_order, goal_tasks):
            self.planning_stats["failed_plans"] += 1
            raise ValueError("Generated plan is invalid")
        
        self.current_plan = execution_order
        self.planning_stats["successful_plans"] += 1
        
        planning_time = time.time() - start_time
        self.planning_stats["average_planning_time"] = (
            (self.planning_stats["average_planning_time"] * (self.planning_stats["successful_plans"] - 1) + planning_time) /
            self.planning_stats["successful_plans"]
        )
        
        self.logger.info(
            "Plan generated",
            goal_id=goal_id,
            plan_length=len(execution_order),
            planning_time=planning_time
        )
        
        return execution_order
    
    def _build_dependency_graph(self, tasks: List[Task]) -> Dict[str, List[str]]:
        """Build dependency graph from tasks"""
        graph = {task.id: [] for task in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in graph:
                    graph[dep_id].append(task.id)
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort to determine execution order"""
        in_degree = {node: 0 for node in graph}
        
        # Calculate in-degrees
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        # Kahn's algorithm
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            # Sort by priority (higher priority first)
            queue.sort(key=lambda x: self.tasks[x].priority, reverse=True)
            
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _validate_plan(self, execution_order: List[str], tasks: List[Task]) -> bool:
        """Validate that the plan is feasible"""
        completed_tasks = set()
        
        for task_id in execution_order:
            task = self.tasks[task_id]
            
            # Check dependencies
            if not task.is_ready(completed_tasks):
                return False
            
            completed_tasks.add(task_id)
        
        return True
    
    def execute_next_task(self) -> Optional[Task]:
        """
        Execute the next task in the current plan
        
        Key Concepts:
        - Plan Execution: Carrying out planned actions
        - Progress Tracking: Monitoring plan execution
        - Outcome Recording: Learning from task outcomes
        """
        
        if not self.current_plan:
            return None
        
        task_id = self.current_plan[0]
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.PENDING:
            # Skip completed or failed tasks
            self.current_plan.pop(0)
            return self.execute_next_task()
        
        # Execute task
        task.status = TaskStatus.IN_PROGRESS
        
        self.logger.info(
            "Executing task",
            task_id=task_id,
            task_name=task.name,
            goal_id=task.goal_id
        )
        
        return task
    
    def complete_task(self, task_id: str, outcome: Dict[str, Any]):
        """
        Mark task as completed with outcome
        
        Key Concepts:
        - Outcome Recording: Storing what happened
        - Progress Updates: Updating goal progress
        - Learning Integration: Using outcomes for future planning
        """
        
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED
        task.add_outcome(outcome)
        
        # Update goal progress
        goal = self.goals[task.goal_id]
        goal_tasks = [t for t in self.tasks.values() if t.goal_id == task.goal_id]
        completed_tasks = [t for t in goal_tasks if t.is_completed()]
        goal.update_progress(len(completed_tasks) / len(goal_tasks))
        
        # Remove from current plan
        if task_id in self.current_plan:
            self.current_plan.remove(task_id)
        
        # Record execution
        self.execution_history.append({
            "task_id": task_id,
            "goal_id": task.goal_id,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store in memory
        create_memory(
            content=f"Task completed: {task.name} - {outcome.get('summary', 'No summary')}",
            memory_type=MemoryType.EPISODIC,
            priority=MemoryPriority.MEDIUM,
            tags=["task_completion", goal.goal_type.value],
            context={"task_id": task_id, "goal_id": task.goal_id, "outcome": outcome}
        )
        
        self.logger.info(
            "Task completed",
            task_id=task_id,
            task_name=task.name,
            goal_progress=goal.progress
        )
    
    def fail_task(self, task_id: str, reason: str):
        """Mark task as failed"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.add_outcome({"status": "failed", "reason": reason})
        
        # Remove from current plan
        if task_id in self.current_plan:
            self.current_plan.remove(task_id)
        
        self.logger.warning(
            "Task failed",
            task_id=task_id,
            task_name=task.name,
            reason=reason
        )
    
    def replan_if_needed(self, current_situation: Dict[str, Any]) -> bool:
        """
        Check if replanning is needed and generate new plan
        
        Key Concepts:
        - Situation Assessment: Evaluating current state
        - Replanning Triggers: When to create new plans
        - Plan Adaptation: Modifying plans based on new information
        """
        
        if not self.current_plan:
            return False
        
        # Check replanning conditions
        needs_replan = False
        
        # Check if current plan is still valid
        for task_id in self.current_plan:
            task = self.tasks[task_id]
            
            # Check if task is still feasible
            if not self._is_task_feasible(task, current_situation):
                needs_replan = True
                break
        
        # Check if goals have changed
        active_goals = [g for g in self.goals.values() if g.is_active()]
        if len(active_goals) > 1:
            # Multiple active goals might need reprioritization
            needs_replan = True
        
        if needs_replan:
            self.planning_stats["replanning_events"] += 1
            
            # Generate new plan for highest priority goal
            active_goals.sort(key=lambda g: g.priority, reverse=True)
            top_goal = active_goals[0]
            
            try:
                new_plan = self.generate_plan(top_goal.id)
                self.logger.info(
                    "Replanning completed",
                    goal_id=top_goal.id,
                    new_plan_length=len(new_plan)
                )
                return True
            except Exception as e:
                self.logger.error("Replanning failed", error=str(e))
                return False
        
        return False
    
    def _is_task_feasible(self, task: Task, situation: Dict[str, Any]) -> bool:
        """Check if task is feasible given current situation"""
        # Check resource availability
        for resource, requirement in task.required_resources.items():
            if resource not in situation.get("resources", {}):
                return False
        
        # Check execution conditions
        for condition in task.execution_conditions:
            if not self._evaluate_condition(condition, situation):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: str, situation: Dict[str, Any]) -> bool:
        """Evaluate execution condition"""
        # Simplified condition evaluation
        # In a real implementation, this would use a more sophisticated parser
        if "health" in condition and "health" in situation:
            return situation["health"] > 0
        if "location" in condition and "location" in situation:
            return condition in situation["location"]
        return True
    
    def get_planning_stats(self) -> Dict[str, Any]:
        """Get comprehensive planning statistics"""
        stats = self.planning_stats.copy()
        
        # Add current state
        stats.update({
            "active_goals": len([g for g in self.goals.values() if g.is_active()]),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "current_plan_length": len(self.current_plan),
            "total_executions": len(self.execution_history)
        })
        
        return stats
    
    def display_current_plan(self):
        """Display current plan in a nice format"""
        if not self.current_plan:
            console.print("No active plan", style="yellow")
            return
        
        table = Table(title="Current Execution Plan")
        table.add_column("Order", style="cyan")
        table.add_column("Task", style="green")
        table.add_column("Goal", style="blue")
        table.add_column("Priority", style="magenta")
        table.add_column("Status", style="yellow")
        
        for i, task_id in enumerate(self.current_plan, 1):
            task = self.tasks[task_id]
            goal = self.goals[task.goal_id]
            
            table.add_row(
                str(i),
                task.name,
                goal.name,
                str(task.priority),
                task.status.value
            )
        
        console.print(table)


# Global planning system instance
planning_system = PlanningSystem()


# Utility functions for easy access
def create_goal(*args, **kwargs) -> Goal:
    """Create a new goal"""
    return planning_system.create_goal(*args, **kwargs)


def create_task(*args, **kwargs) -> Task:
    """Create a new task"""
    return planning_system.create_task(*args, **kwargs)


def generate_plan(*args, **kwargs) -> List[str]:
    """Generate execution plan"""
    return planning_system.generate_plan(*args, **kwargs)


def get_planning_stats() -> Dict[str, Any]:
    """Get planning statistics"""
    return planning_system.get_planning_stats() 