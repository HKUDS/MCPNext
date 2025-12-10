import asyncio
import argparse
import sys
import logging
from typing import Optional

from mcpnext.tool_layer import MCPNext, MCPNextConfig
from mcpnext.utils.logging import Logger
from mcpnext.utils.ui import create_ui, MCPNextUI
from mcpnext.utils.ui_integration import UIIntegration
from mcpnext.utils.cli_display import CLIDisplay
from mcpnext.utils.display import colorize

logger = Logger.get_logger(__name__)


class UIManager:
    def __init__(self, ui: Optional[MCPNextUI], ui_integration: Optional[UIIntegration]):
        self.ui = ui
        self.ui_integration = ui_integration
        self._original_log_levels = {}
    
    async def start_live_display(self):
        if not self.ui or not self.ui_integration:
            return
        
        print()
        print(colorize("  ▣ Starting real-time visualization...", 'c'))
        print()
        await asyncio.sleep(1)
        
        self._suppress_logs()
        
        await self.ui.start_live_display()
        await self.ui_integration.start_monitoring(poll_interval=2.0)
    
    async def stop_live_display(self):
        if not self.ui or not self.ui_integration:
            return
        
        await self.ui_integration.stop_monitoring()
        await self.ui.stop_live_display()
        
        self._restore_logs()
    
    def print_summary(self, result: dict):
        if self.ui:
            self.ui.print_summary(result)
        else:
            CLIDisplay.print_result_summary(result)
    
    def _suppress_logs(self):
        log_names = ["mcpnext", "mcpnext.grounding", "mcpnext.agents"]
        for name in log_names:
            log = logging.getLogger(name)
            self._original_log_levels[name] = log.level
            log.setLevel(logging.CRITICAL)
    
    def _restore_logs(self):
        for name, level in self._original_log_levels.items():
            logging.getLogger(name).setLevel(level)
        self._original_log_levels.clear()


async def _execute_task(mcpnext: MCPNext, query: str, ui_manager: UIManager):
    await ui_manager.start_live_display()
    result = await mcpnext.execute(query)
    await ui_manager.stop_live_display()
    ui_manager.print_summary(result)
    return result


async def interactive_mode(mcpnext: MCPNext, ui_manager: UIManager):
    CLIDisplay.print_interactive_header()
    
    while True:
        try:
            prompt = colorize(">>> ", 'c', bold=True)
            query = input(f"\n{prompt}").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nExiting...")
                break

            if query.lower() == 'status':
                _print_status(mcpnext)
                continue
            
            if query.lower() == 'help':
                CLIDisplay.print_help()
                continue

            CLIDisplay.print_task_header(query)
            await _execute_task(mcpnext, query, ui_manager)
            
        except KeyboardInterrupt:
            print("\n\nInterrupt signal detected, exiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"\nError: {e}")


async def single_query_mode(mcpnext: MCPNext, query: str, ui_manager: UIManager):
    CLIDisplay.print_task_header(query, title="▶ Single Query Execution")
    await _execute_task(mcpnext, query, ui_manager)


def _print_status(mcpnext: MCPNext):
    """Print system status"""
    from mcpnext.utils.display import Box, BoxStyle
    
    box = Box(width=70, style=BoxStyle.ROUNDED, color='bl')
    print()
    print(box.text_line(colorize("System Status", 'bl', bold=True), 
                      align='center', indent=4, text_color=''))
    print(box.separator_line(indent=4))
    
    status_lines = [
        f"Initialized: {colorize('Yes' if mcpnext.is_initialized() else 'No', 'g' if mcpnext.is_initialized() else 'rd')}",
        f"Running: {colorize('Yes' if mcpnext.is_running() else 'No', 'y' if mcpnext.is_running() else 'g')}",
        f"Model: {colorize(mcpnext.config.llm_model, 'c')}",
    ]
    
    if mcpnext.is_initialized():
        backends = mcpnext.list_backends()
        status_lines.append(f"Backends: {colorize(', '.join(backends), 'c')}")
        
        sessions = mcpnext.list_sessions()
        status_lines.append(f"Active Sessions: {colorize(str(len(sessions)), 'y')}")
    
    for line in status_lines:
        print(box.text_line(f"  {line}", indent=4, text_color=''))
    
    print(box.bottom_line(indent=4))
    print()


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='MCPNext - Universal Tool-Use Layer for AI Agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Basic arguments
    parser.add_argument('--config', '-c', type=str, help='Configuration file path (JSON format)')
    parser.add_argument('--query', '-q', type=str, help='Single query mode: execute query directly')
    
    # LLM arguments
    parser.add_argument('--model', '-m', type=str, help='LLM model name')
    
    # Logging arguments
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    # Execution arguments
    parser.add_argument('--max-iterations', type=int, help='Maximum iteration count')
    parser.add_argument('--timeout', type=float, help='LLM API call timeout (seconds)')
    
    # UI arguments
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive mode')
    parser.add_argument('--no-ui', action='store_true', help='Disable visualization UI')
    parser.add_argument('--ui-compact', action='store_true', help='Use compact UI layout')
    
    return parser


def _load_config(args) -> MCPNextConfig:
    """Load configuration"""
    cli_overrides = {}
    if args.model:
        cli_overrides['llm_model'] = args.model
    if args.max_iterations is not None:
        cli_overrides['grounding_max_iterations'] = args.max_iterations
    if args.timeout is not None:
        cli_overrides['llm_timeout'] = args.timeout
    if args.log_level:
        cli_overrides['log_level'] = args.log_level
    
    try:
        # Load from config file if provided
        if args.config:
            import json
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Apply CLI overrides
            config_dict.update(cli_overrides)
            config = MCPNextConfig(**config_dict)
            
            print(f"✓ Loaded from config file: {args.config}")
        else:
            # Use default config + CLI overrides
            config = MCPNextConfig(**cli_overrides)
            print("✓ Using default configuration")
        
        if cli_overrides:
            print(f"✓ CLI overrides: {', '.join(cli_overrides.keys())}")
        
        if args.log_level:
            Logger.set_level(args.log_level)
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def _setup_ui(args) -> tuple[Optional[MCPNextUI], Optional[UIIntegration]]:
    if args.no_ui:
        CLIDisplay.print_banner()
        return None, None
    
    ui = create_ui(enable_live=True, compact=args.ui_compact)
    ui.print_banner()
    ui_integration = UIIntegration(ui)
    return ui, ui_integration


async def _initialize_mcpnext(config: MCPNextConfig, args) -> MCPNext:
    mcpnext = MCPNext(config)
    
    init_steps = [("Initializing MCPNext...", "loading")]
    CLIDisplay.print_initialization_progress(init_steps, show_header=False)
    
    if not args.config:
        original_log_level = Logger.get_logger("mcpnext").level
        for log_name in ["mcpnext", "mcpnext.grounding", "mcpnext.agents"]:
            Logger.get_logger(log_name).setLevel(logging.WARNING)
    
    await mcpnext.initialize()
    
    # Restore log level
    if not args.config:
        for log_name in ["mcpnext", "mcpnext.grounding", "mcpnext.agents"]:
            Logger.get_logger(log_name).setLevel(original_log_level)
    
    # Print initialization results
    backends = mcpnext.list_backends()
    init_steps = [
        ("LLM Client", "ok"),
        (f"Grounding Backends ({len(backends)} available)", "ok"),
        ("Grounding Agent", "ok"),
    ]
    
    if config.enable_recording:
        init_steps.append(("Recording Manager", "ok"))
    
    CLIDisplay.print_initialization_progress(init_steps, show_header=True)
    
    return mcpnext


async def main():
    parser = _create_argument_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = _load_config(args)
    
    # Setup UI
    ui, ui_integration = _setup_ui(args)
    
    # Print configuration
    CLIDisplay.print_configuration(config)
    
    mcpnext = None
    
    try:
        # Initialize MCPNext
        mcpnext = await _initialize_mcpnext(config, args)
        
        # Connect UI (if enabled)
        if ui_integration:
            ui_integration.attach_llm_client(mcpnext._llm_client)
            ui_integration.attach_grounding_client(mcpnext._grounding_client)
            CLIDisplay.print_system_ready()
        
        ui_manager = UIManager(ui, ui_integration)
        
        # Run appropriate mode
        if args.query:
            await single_query_mode(mcpnext, args.query, ui_manager)
        else:
            await interactive_mode(mcpnext, ui_manager)
        
    except KeyboardInterrupt:
        print("\n\nInterrupt signal detected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    finally:
        if mcpnext:
            print("\nCleaning up resources...")
            await mcpnext.cleanup()
    
    print("\nGoodbye!")
    return 0


def run_main():
    """Run main function"""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted")
        sys.exit(0)


if __name__ == "__main__":
    run_main()