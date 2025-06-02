import argparse
import logging
import pathlib
import sys
import os

from rich.console import Console
from rich.markdown import Markdown

# Imports from within the buddy_ai package
from .config import load_app_config
from .utils import read_file_or_directory
from .agent import (
    create_llm_instance,
    create_executor_agent_runnable,
    set_global_llms_and_agents,
    create_buddy_graph,
    set_agent_console
)

# This console is specific to the CLI execution context.
# The agent itself might use a different console if set by other means.
_cli_execution_console = Console()

def setup_logging():
    # Consider making log level configurable via CLI argument as well
    log_level_str = os.getenv("BUDDY_LOG_LEVEL", "WARNING").upper()
    # Ensure the log level is a valid one
    log_level = getattr(logging, log_level_str, None)
    if not isinstance(log_level, int):
        print(f"Warning: Invalid BUDDY_LOG_LEVEL '{log_level_str}'. Defaulting to WARNING.")
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Use __name__ for the logger to reflect the module it's in
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized with level {logging.getLevelName(log_level)} for {__name__}.")

def main():
    setup_logging()
    logger = logging.getLogger(__name__) # Get a logger for main module

    # Set the console for the agent module (used by human_approval_node etc.)
    # This uses the _cli_execution_console created in this __main__ scope.
    set_agent_console(_cli_execution_console)

    parser = argparse.ArgumentParser(description="Buddy AI Agent CLI - Your AI assistant for various tasks.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="User objective, instructions, or a path to a text file containing them.")
    parser.add_argument("--context", type=str,
                        help="Path to a file or directory to provide as context to the agent.")
    parser.add_argument("--auto", action="store_true", help="Enable auto mode, bypassing human plan approval.")
    # Potentially add --log-level argument here in the future
    args = parser.parse_args()

    logger.info("Loading application configuration...")
    api_key, planner_model_name, executor_model_name = load_app_config()

    if not api_key:
        _cli_execution_console.print(Markdown("**CRITICAL ERROR:** API key not found. Ensure `config.ini` is in CWD and configured, or environment variables are set."))
        sys.exit("CRITICAL: API key config error.")
    if not planner_model_name or not executor_model_name:
        _cli_execution_console.print(Markdown("**CRITICAL ERROR:** Model names not loaded. Check `config.ini` or environment variables."))
        sys.exit("CRITICAL: Model name config error.")

    prompt_input = args.prompt
    prompt_source_msg = "User objective"
    if pathlib.Path(prompt_input).is_file():
        try:
            prompt_input = pathlib.Path(prompt_input).read_text(encoding='utf-8')
            logger.info(f"Prompt loaded from file: {args.prompt}")
            prompt_source_msg = f"User objective (from file: {args.prompt})"
        except Exception as e:
            logger.error(f"Error reading prompt file {args.prompt}: {e}", exc_info=True)
            _cli_execution_console.print(Markdown(f"**Critical Error:** Could not read prompt file `{args.prompt}`."))
            sys.exit(1)

    context_input_str = ""
    if args.context:
        logger.info(f"Loading context from: {args.context}")
        context_input_str = read_file_or_directory(args.context)
        if context_input_str.startswith("Error:"):
             _cli_execution_console.print(Markdown(f"**Warning:** Could not load context: {context_input_str}"))

    if args.context and context_input_str.startswith("Error:"):
        _cli_execution_console.print(Markdown(f"--- \n*Failed to load context from {args.context}.*"))
    
    logger.info("Initializing LLMs and Agent...")
    planner_llm = create_llm_instance(planner_model_name, "Planner", api_key)
    if not planner_llm:
        _cli_execution_console.print(Markdown("**CRITICAL ERROR:** Planner LLM failed to initialize."))
        sys.exit("CRITICAL: Planner LLM init failed.")

    executor_llm = create_llm_instance(executor_model_name, "Executor", api_key)
    if not executor_llm:
        _cli_execution_console.print(Markdown("**CRITICAL ERROR:** Executor LLM failed to initialize."))
        sys.exit("CRITICAL: Executor LLM init failed.")

    executor_agent_runnable = create_executor_agent_runnable(executor_llm)
    if not executor_agent_runnable:
        _cli_execution_console.print(Markdown("**CRITICAL ERROR:** Executor ReAct agent failed to create."))
        sys.exit("CRITICAL: Executor agent creation failed.")

    try:
        set_global_llms_and_agents(planner_llm, executor_agent_runnable)
    except Exception as e:
        _cli_execution_console.print(Markdown(f"**CRITICAL ERROR:** Failed to set global LLMs/agents: {e}"))
        logger.critical(f"Failed to set global LLMs/agents: {e}", exc_info=True)
        sys.exit("CRITICAL: Global LLM/agent setup failed.")

    buddy_app = create_buddy_graph()
    initial_state = {
        "objective": prompt_input,
        "context": context_input_str if context_input_str and not context_input_str.startswith("Error:") else "",
        "plan": None,
        "current_step_index": 0,
        "step_results": [],
        "final_output": None,
        "user_feedback": None,
        "plan_approved": False,
        "auto_approve": args.auto
    }

    _cli_execution_console.print(Markdown("\n# --- Buddy AI Workflow Starting ---"))
    logger.info("Invoking StateGraph workflow.")

    # Accumulate state here as stream() gives partials
    current_graph_state_accumulator = initial_state.copy()

    try:
        _cli_execution_console.print(Markdown("Generating execution plan..."))
        for event in buddy_app.stream(initial_state, {"recursion_limit": 25}):
            logger.debug(f"Graph event: {event}")
            # Correctly merge event data into the accumulator
            for node_name, node_output_update in event.items():
                # node_output_update is a dict of state keys this node has changed
                for state_key, new_value in node_output_update.items():
                    current_graph_state_accumulator[state_key] = new_value

                # Process specific node events for CLI output
                if node_name == "planner":
                    current_plan = current_graph_state_accumulator.get("plan")
                    if current_plan and isinstance(current_plan, list) and current_plan[0].startswith("Critical Error:"):
                        _cli_execution_console.print(Markdown(f"**PLANNING FAILED:** {current_plan[0]}"))

                elif node_name == "human_approval":
                    # Use accumulated state for these checks
                    auto_approve = current_graph_state_accumulator.get("auto_approve", False)
                    plan_approved = current_graph_state_accumulator.get("plan_approved", False)
                    user_feedback = current_graph_state_accumulator.get("user_feedback")
                    if not auto_approve and not plan_approved and not user_feedback:
                        _cli_execution_console.print(Markdown("--- \n*Waiting for human approval (Approve/Refine/Cancel)...*"))
                    elif auto_approve and plan_approved:
                        _cli_execution_console.print(Markdown("--- \n*Plan auto-approved.*"))

                elif node_name == "executor":
                    executed_step_idx = current_graph_state_accumulator.get("current_step_index", 1) - 1
                    full_plan = current_graph_state_accumulator.get("plan", [])
                    plan_len = len(full_plan) if full_plan is not None else 0 # Handle if plan is None

                    step_instruction = "Unknown step"
                    if full_plan and 0 <= executed_step_idx < plan_len:
                         step_instruction = full_plan[executed_step_idx]

                    _cli_execution_console.print(Markdown(f"### Executing Step {executed_step_idx + 1}/{plan_len}: *{step_instruction}*"))

                    step_results_list = current_graph_state_accumulator.get("step_results", [])
                    step_result = step_results_list[-1] if step_results_list else "No result recorded."
                    _cli_execution_console.print(Markdown(f"**Result of Step {executed_step_idx + 1}:**\n```text\n{step_result}\n```"))

                    if isinstance(step_result, str) and step_result.startswith("Critical Error:"):
                        _cli_execution_console.print(Markdown(f"**EXECUTION FAILED at step {executed_step_idx+1}.**"))

    except Exception as e:
        logger.error(f"Error during graph streaming: {e}", exc_info=True)
        _cli_execution_console.print(Markdown(f"\n**CRITICAL ERROR during graph execution:** {e}. Check logs."))
        current_graph_state_accumulator["final_output"] = f"CRITICAL ERROR during graph execution: {e}"


    _cli_execution_console.print(Markdown("\n# --- Buddy AI Workflow Complete ---"))

    final_plan = current_graph_state_accumulator.get('plan', [])
    plan_approved = current_graph_state_accumulator.get('plan_approved', False)
    user_feedback = current_graph_state_accumulator.get('user_feedback')
    auto_approved_mode = current_graph_state_accumulator.get('auto_approve', False) # from initial state

    successful_run = False

    if final_plan and isinstance(final_plan, list) and len(final_plan) > 0 and final_plan[0].startswith("Critical Error:"):
        _cli_execution_console.print(Markdown(f"**Workflow ended due to PLANNER error:** {final_plan[0]}"))
    elif not plan_approved and not user_feedback and not auto_approved_mode :
         _cli_execution_console.print(Markdown("**Workflow ended: Plan was not approved and no feedback was provided (cancelled).**"))
    else:
        final_step_results = current_graph_state_accumulator.get('step_results', [])
        if final_step_results:
            last_result = str(final_step_results[-1]) if final_step_results else ""
            if last_result.startswith("Critical Error:"):
                 _cli_execution_console.print(Markdown(f"**Workflow ended due to EXECUTOR error:** {last_result}"))
            elif plan_approved or auto_approved_mode: # If plan was approved (manually or auto) and executed
                _cli_execution_console.print(Markdown("## --- Consolidated Output from All Steps ---"))
                str_step_results = [str(res) for res in final_step_results]
                final_consolidated_output = "\n\n---\n\n".join(str_step_results)
                _cli_execution_console.print(Markdown(final_consolidated_output))
                if not last_result.startswith("Error:"): # Consider it success if last step wasn't an "Error:" (but not "Critical Error:")
                    successful_run = True
            elif user_feedback: # Plan was refined, loop would continue in a real scenario
                 _cli_execution_console.print(Markdown("**Workflow cycle ended after user feedback for replan. Run again with new plan if generated.**"))
        elif plan_approved or auto_approved_mode: # Approved but no steps executed or results recorded
            _cli_execution_console.print(Markdown("*Plan was approved, but no step results were recorded or steps were executed.*"))
            successful_run = True # If plan was just to, e.g., confirm something without shell commands
        elif not plan_approved and not user_feedback and auto_approved_mode: # Should not happen if auto_approve is true
             _cli_execution_console.print(Markdown("*Workflow ended: Plan was not auto-approved despite --auto flag and no steps executed. Check agent logic.*"))
        else: # Other unhandled cases
            _cli_execution_console.print(Markdown("*Workflow ended for other reasons or with no clear output.*"))

    logger.info("Buddy application finished.")

    if successful_run:
        _cli_execution_console.print(Markdown("--- \n**Workflow concluded (check output for operational success).**"))
        sys.exit(0)
    else:
        _cli_execution_console.print(Markdown("--- \n**Workflow concluded with errors, was cancelled, or requires review.**"))
        sys.exit(1)

if __name__ == "__main__":
    main()
