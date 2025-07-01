"""
Main Entry Point - Agentic AI Dungeon Lord
==========================================

This is the main entry point for the Agentic AI Dungeon Lord system.
It demonstrates the complete agentic AI architecture in action.

Key Features:
1. **Complete System Integration**: All subsystems working together
2. **Interactive Gameplay**: Real-time interaction with the agentic AI
3. **Performance Monitoring**: Real-time statistics and insights
4. **Learning Demonstration**: See the AI learn and adapt

Usage:
    python main.py

This will start an interactive D&D session with a fully agentic AI Dungeon Master.
"""

import os
import sys
import time
import signal
from typing import Optional
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from dotenv import load_dotenv

# Import our agentic AI system
from agentic_agent import process_input, get_agent_stats, display_agent_status
from config import get_config, reload_config
from memory_system import get_memory_stats
from planning_system import get_planning_stats
from tools_system import get_tool_stats, list_tools

# Load environment variables
load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()


class AgenticDungeonLord:
    """
    Main Application Class - Orchestrates the Complete Agentic AI Experience
    
    This class provides the main interface for interacting with the agentic AI system.
    It demonstrates all the key AI/ML concepts in a user-friendly way.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(component="main_application")
        
        # Application state
        self.running = False
        self.session_start_time = None
        self.interaction_count = 0
        
        # Performance tracking
        self.session_stats = {
            "total_interactions": 0,
            "average_response_time": 0.0,
            "system_uptime": 0.0,
            "memory_usage": 0.0,
            "tool_usage": 0
        }
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Agentic Dungeon Lord application initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully")
        self.shutdown()
        sys.exit(0)
    
    def startup(self):
        """Start the application and display welcome screen"""
        
        console.clear()
        
        # Display welcome screen
        welcome_text = """
🤖 AGENTIC AI DUNGEON LORD 🤖
===============================

Welcome to the future of AI-powered D&D gameplay!

This system demonstrates cutting-edge AI/ML concepts:
• RAG (Retrieval-Augmented Generation)
• ReAct Pattern (Reasoning + Acting)
• Hierarchical Planning
• Multi-Agent Coordination
• Continuous Learning & Adaptation
• Memory Management & Consolidation

Your AI Dungeon Master is equipped with:
• Semantic Memory System
• Goal-Directed Planning
• Tool Integration (Dice, Combat, NPCs)
• Self-Reflection & Learning
• Performance Optimization

Ready to experience the next generation of AI gaming?
        """
        
        console.print(Panel(welcome_text, title="Welcome to Agentic AI", border_style="blue"))
        
        # Check configuration
        self._check_configuration()
        
        # Initialize session
        self.session_start_time = time.time()
        self.running = True
        
        # Display initial status
        self._display_initial_status()
        
        self.logger.info("Application started successfully")
    
    def _check_configuration(self):
        """Check and validate configuration"""
        
        console.print("\n🔧 Checking Configuration...")
        
        # Check API keys
        if not self.config.llm.openai_api_key:
            console.print("[red]❌ OpenAI API key not found![/red]")
            console.print("Please set the OPENAI_API_KEY environment variable.")
            sys.exit(1)
        
        # Check required directories
        required_dirs = [
            self.config.memory.vector_db_path,
            "logs",
            "data"
        ]
        
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
        
        console.print("[green]✅ Configuration validated[/green]")
    
    def _display_initial_status(self):
        """Display initial system status"""
        
        console.print("\n📊 Initial System Status:")
        
        # Get subsystem stats
        memory_stats = get_memory_stats()
        planning_stats = get_planning_stats()
        tools_stats = get_tool_stats()
        
        # Create status table
        table = Table(title="System Status")
        table.add_column("Subsystem", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        table.add_row("Memory System", "✅ Active", f"{memory_stats['short_term_count']} short-term, {sum(memory_stats['collections'].values())} long-term memories")
        table.add_row("Planning System", "✅ Active", f"{planning_stats['active_goals']} active goals, {planning_stats['pending_tasks']} pending tasks")
        table.add_row("Tools System", "✅ Active", f"{tools_stats['available_tools']} tools available, {tools_stats['total_executions']} executions")
        table.add_row("Agentic Agent", "✅ Active", "Ready for interaction")
        
        console.print(table)
    
    def run(self):
        """Main application loop"""
        
        console.print("\n🎮 Starting Interactive Session...")
        console.print("Type 'help' for commands, 'quit' to exit\n")
        
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_detailed_stats()
                    continue
                elif user_input.lower() == 'tools':
                    self._show_tools()
                    continue
                elif user_input.lower() == 'memory':
                    self._show_memory_info()
                    continue
                elif user_input.lower() == 'planning':
                    self._show_planning_info()
                    continue
                elif user_input.lower() == 'demo':
                    self._run_demo()
                    continue
                elif user_input.strip() == '':
                    continue
                
                # Process input through agentic AI
                start_time = time.time()
                
                console.print("\n[bold green]🤖 AI Dungeon Master[/bold green]")
                with console.status("[bold green]Thinking...", spinner="dots"):
                    response = process_input(user_input)
                
                response_time = time.time() - start_time
                
                # Display response
                console.print(f"\n{response}\n")
                
                # Update session stats
                self._update_session_stats(response_time)
                
                # Show performance indicator
                self._show_performance_indicator(response_time)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted by user[/yellow]")
                break
            except Exception as e:
                self.logger.error("Error in main loop", error=str(e))
                console.print(f"\n[red]Error: {str(e)}[/red]")
    
    def _show_help(self):
        """Show help information"""
        
        help_text = """
🎮 Available Commands:
====================

Game Commands:
• Any text - Interact with the AI Dungeon Master
• Type naturally as you would with a human DM

System Commands:
• help - Show this help message
• status - Show current system status
• stats - Show detailed statistics
• tools - Show available tools
• memory - Show memory system info
• planning - Show planning system info
• demo - Run a demonstration
• quit/exit/q - Exit the application

AI/ML Concepts Demonstrated:
• RAG (Retrieval-Augmented Generation)
• ReAct Pattern (Reasoning + Acting)
• Hierarchical Planning
• Memory Consolidation
• Tool Integration
• Self-Reflection & Learning
        """
        
        console.print(Panel(help_text, title="Help", border_style="green"))
    
    def _show_status(self):
        """Show current system status"""
        
        display_agent_status()
    
    def _show_detailed_stats(self):
        """Show detailed system statistics"""
        
        console.print("\n📈 Detailed System Statistics:")
        
        # Get all stats
        agent_stats = get_agent_stats()
        memory_stats = get_memory_stats()
        planning_stats = get_planning_stats()
        tools_stats = get_tool_stats()
        
        # Create comprehensive stats table
        table = Table(title="Comprehensive System Statistics")
        table.add_column("Category", style="cyan")
        table.add_column("Metric", style="green")
        table.add_column("Value", style="yellow")
        
        # Agent stats
        table.add_row("Agent", "State", agent_stats["agent_state"])
        table.add_row("Agent", "Mode", agent_stats["agent_mode"])
        table.add_row("Agent", "Total Interactions", str(agent_stats["performance_metrics"]["total_interactions"]))
        table.add_row("Agent", "Avg Response Time", f"{agent_stats['performance_metrics']['average_response_time']:.2f}s")
        table.add_row("Agent", "Success Rate", f"{agent_stats['performance_metrics']['goal_completion_rate']:.1%}")
        
        # Memory stats
        table.add_row("Memory", "Short-term", str(memory_stats["short_term_count"]))
        table.add_row("Memory", "Hit Rate", f"{memory_stats['access_stats']['memory_hit_rate']:.1%}")
        table.add_row("Memory", "Avg Similarity", f"{memory_stats['access_stats']['average_similarity']:.2f}")
        
        # Planning stats
        table.add_row("Planning", "Active Goals", str(planning_stats["active_goals"]))
        table.add_row("Planning", "Pending Tasks", str(planning_stats["pending_tasks"]))
        table.add_row("Planning", "Success Rate", f"{(planning_stats['successful_plans'] / max(planning_stats['successful_plans'] + planning_stats['failed_plans'], 1) * 100):.1f}%")
        
        # Tools stats
        table.add_row("Tools", "Available", str(tools_stats["available_tools"]))
        table.add_row("Tools", "Success Rate", f"{(tools_stats['successful_executions'] / max(tools_stats['total_executions'], 1) * 100):.1f}%")
        table.add_row("Tools", "Avg Execution Time", f"{tools_stats['average_execution_time']:.3f}s")
        
        console.print(table)
    
    def _show_tools(self):
        """Show available tools"""
        
        tools = list_tools()
        
        console.print("\n🔧 Available Tools:")
        
        table = Table(title="Tool Registry")
        table.add_column("Tool Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Usage Count", style="magenta")
        table.add_column("Status", style="blue")
        
        for tool in tools:
            status = "✅ Enabled" if tool["enabled"] else "❌ Disabled"
            table.add_row(
                tool["name"],
                tool["type"],
                tool["description"][:50] + "..." if len(tool["description"]) > 50 else tool["description"],
                str(tool["usage_count"]),
                status
            )
        
        console.print(table)
    
    def _show_memory_info(self):
        """Show memory system information"""
        
        memory_stats = get_memory_stats()
        
        console.print("\n🧠 Memory System Information:")
        
        # Memory collections
        table = Table(title="Memory Collections")
        table.add_column("Memory Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Description", style="yellow")
        
        memory_types = {
            "episodic": "Events and experiences",
            "semantic": "Facts and knowledge",
            "procedural": "Skills and procedures",
            "emotional": "Emotional associations",
            "spatial": "Location and navigation"
        }
        
        for memory_type, description in memory_types.items():
            count = memory_stats["collections"].get(memory_type, 0)
            table.add_row(memory_type.title(), str(count), description)
        
        console.print(table)
        
        # Access statistics
        access_stats = memory_stats["access_stats"]
        console.print(f"\n📊 Memory Access Statistics:")
        console.print(f"• Total Retrievals: {access_stats['total_retrievals']}")
        console.print(f"• Successful Retrievals: {access_stats['successful_retrievals']}")
        console.print(f"• Hit Rate: {access_stats['memory_hit_rate']:.1%}")
        console.print(f"• Average Similarity: {access_stats['average_similarity']:.2f}")
    
    def _show_planning_info(self):
        """Show planning system information"""
        
        planning_stats = get_planning_stats()
        
        console.print("\n🎯 Planning System Information:")
        
        # Planning statistics
        table = Table(title="Planning Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Description", style="yellow")
        
        table.add_row("Active Goals", str(planning_stats["active_goals"]), "Currently pursued goals")
        table.add_row("Pending Tasks", str(planning_stats["pending_tasks"]), "Tasks waiting to be executed")
        table.add_row("Successful Plans", str(planning_stats["successful_plans"]), "Plans that were successfully generated")
        table.add_row("Failed Plans", str(planning_stats["failed_plans"]), "Plans that failed to generate")
        table.add_row("Replanning Events", str(planning_stats["replanning_events"]), "Times the system had to replan")
        table.add_row("Avg Planning Time", f"{planning_stats['average_planning_time']:.3f}s", "Average time to generate plans")
        
        console.print(table)
    
    def _run_demo(self):
        """Run a demonstration of the agentic AI capabilities"""
        
        console.print("\n🎬 Running Agentic AI Demonstration...")
        
        demo_commands = [
            "I want to explore the dungeon",
            "Roll a d20 for perception",
            "I encounter a goblin, what do I do?",
            "Generate an NPC merchant",
            "What's in my inventory?",
            "I want to plan my next move carefully"
        ]
        
        for i, command in enumerate(demo_commands, 1):
            console.print(f"\n[bold cyan]Demo Step {i}:[/bold cyan] {command}")
            
            # Process command
            start_time = time.time()
            response = process_input(command)
            response_time = time.time() - start_time
            
            console.print(f"[bold green]AI Response:[/bold green] {response}")
            console.print(f"[dim]Response time: {response_time:.2f}s[/dim]")
            
            # Pause between commands
            if i < len(demo_commands):
                time.sleep(2)
        
        console.print("\n[green]✅ Demonstration completed![/green]")
        console.print("This demonstrates the agentic AI's capabilities including:")
        console.print("• Natural language understanding")
        console.print("• Tool integration (dice rolling, NPC generation)")
        console.print("• Planning and goal-directed behavior")
        console.print("• Memory retrieval and context awareness")
    
    def _update_session_stats(self, response_time: float):
        """Update session statistics"""
        
        self.session_stats["total_interactions"] += 1
        self.session_stats["system_uptime"] = time.time() - self.session_start_time
        
        # Update average response time
        current_avg = self.session_stats["average_response_time"]
        total_interactions = self.session_stats["total_interactions"]
        self.session_stats["average_response_time"] = (
            (current_avg * (total_interactions - 1) + response_time) / total_interactions
        )
    
    def _show_performance_indicator(self, response_time: float):
        """Show performance indicator"""
        
        if response_time < 1.0:
            indicator = "[green]⚡ Fast[/green]"
        elif response_time < 3.0:
            indicator = "[yellow]🐌 Moderate[/yellow]"
        else:
            indicator = "[red]🐌 Slow[/red]"
        
        console.print(f"[dim]Performance: {indicator} ({response_time:.2f}s)[/dim]")
    
    def shutdown(self):
        """Shutdown the application gracefully"""
        
        if not self.running:
            return
        
        self.running = False
        
        console.print("\n🔄 Shutting down gracefully...")
        
        # Display final statistics
        if self.session_start_time:
            uptime = time.time() - self.session_start_time
            console.print(f"\n📊 Session Summary:")
            console.print(f"• Total Interactions: {self.session_stats['total_interactions']}")
            console.print(f"• Average Response Time: {self.session_stats['average_response_time']:.2f}s")
            console.print(f"• System Uptime: {uptime:.1f}s")
        
        # Final status display
        display_agent_status()
        
        console.print("\n[green]✅ Shutdown complete. Thank you for using Agentic AI Dungeon Lord![/green]")
        
        self.logger.info("Application shutdown complete")


def main():
    """Main entry point"""
    
    try:
        # Create and run the application
        app = AgenticDungeonLord()
        app.startup()
        app.run()
        app.shutdown()
        
    except Exception as e:
        logger.error("Fatal error in main", error=str(e))
        console.print(f"\n[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main() 