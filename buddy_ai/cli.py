import argparse
import logging
import pathlib
import sys
import os # Added for BUDDY_LOG_LEVEL

from rich.console import Console
from rich.markdown import Markdown

from .config import load_app_config
from .utils import read_file_or_directory
from .agent import (
    create_llm_instance,
    create_executor_agent_runnable,
    set_global_llms_and_agents,
    create_buddy_graph
)

cli_console = Console()

def setup_logging():
    # It's good practice to ensure os is imported if using os.getenv
    # import os
    log_level_str = os.getenv("BUDDY_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_str, logging.INFO),
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging initialized with level {log_level_str}.")

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Buddy AI Agent CLI - Your AI assistant for various tasks.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="User objective, instructions, or a path to a text file containing them.")
    parser.add_argument("--context", type=str,
                        help="Path to a file or directory to provide as context to the agent.")
    args = parser.parse_args()

    logging.info("Loading application configuration...")
    api_key, planner_model_name, executor_model_name = load_app_config()

    if not api_key:
        cli_console.print(Markdown("**CRITICAL ERROR:** API key not found. Ensure `config.ini` is in CWD and configured."))
        sys.exit("CRITICAL: API key config error.")
    if not planner_model_name or not executor_model_name:
        cli_console.print(Markdown("**CRITICAL ERROR:** Model names not loaded. Check `config.ini`."))
        sys.exit("CRITICAL: Model name config error.")

    prompt_input = args.prompt
    prompt_source_msg = "User objective"
    if pathlib.Path(prompt_input).is_file():
        try:
            prompt_input = pathlib.Path(prompt_input).read_text(encoding='utf-8')
            logging.info(f"Prompt loaded from file: {args.prompt}")
            prompt_source_msg = f"User objective (from file: {args.prompt})"
        except Exception as e:
            logging.error(f"Error reading prompt file {args.prompt}: {e}", exc_info=True)
            cli_console.print(Markdown(f"**Critical Error:** Could not read prompt file `{args.prompt}`."))
            sys.exit(1)

    context_input_str = ""
    if args.context:
        logging.info(f"Loading context from: {args.context}")
        context_input_str = read_file_or_directory(args.context)
        if context_input_str.startswith("Error:"):
             cli_console.print(Markdown(f"**Warning:** Could not load context: {context_input_str}"))

    cli_console.print(Markdown(f"# {prompt_source_msg}\n\n{prompt_input}"))
    if context_input_str and not context_input_str.startswith("Error:"):
        cli_console.print(Markdown(f"## Context Provided\n{context_input_str}"))
    elif args.context and context_input_str.startswith("Error:"):
        cli_console.print(Markdown(f"--- \n*Failed to load context from {args.context}.*"))
    else:
        cli_console.print(Markdown("--- \n*No context provided.*"))

    logging.info("Initializing LLMs and Agent...")
    planner_llm = create_llm_instance(planner_model_name, "Planner", api_key)
    if not planner_llm:
        cli_console.print(Markdown("**CRITICAL ERROR:** Planner LLM failed to initialize."))
        sys.exit("CRITICAL: Planner LLM init failed.")

    executor_llm = create_llm_instance(executor_model_name, "Executor", api_key)
    if not executor_llm:
        cli_console.print(Markdown("**CRITICAL ERROR:** Executor LLM failed to initialize."))
        sys.exit("CRITICAL: Executor LLM init failed.")

    executor_agent_runnable = create_executor_agent_runnable(executor_llm)
    if not executor_agent_runnable:
        cli_console.print(Markdown("**CRITICAL ERROR:** Executor ReAct agent failed to create."))
        sys.exit("CRITICAL: Executor agent creation failed.")

    try:
        set_global_llms_and_agents(planner_llm, executor_agent_runnable)
    except Exception as e:
        cli_console.print(Markdown(f"**CRITICAL ERROR:** Failed to set global LLMs/agents: {e}"))
        sys.exit("CRITICAL: Global LLM/agent setup failed.")

    buddy_app = create_buddy_graph()
    initial_state = {
        "objective": prompt_input,
        "context": context_input_str if context_input_str and not context_input_str.startswith("Error:") else "",
        "plan": None, "current_step_index": 0, "step_results": [], "final_output": None
    }

    cli_console.print(Markdown("\n# --- Buddy AI Workflow Starting ---"))
    logging.info("Invoking StateGraph workflow.")
    final_graph_state = None # Initialize before try block

    try:
        cli_console.print(Markdown("Generating execution plan..."))
        # Stream the graph execution to display intermediate steps
        # The event structure from app.stream is a dictionary where keys are node names
        # and values are the outputs of those nodes.
        for event in buddy_app.stream(initial_state, {"recursion_limit": 25}):
            if "planner" in event:
                current_plan = event["planner"].get("plan")
                # Update our copy of the plan for executor display
                initial_state["plan"] = current_plan
                if current_plan and not (isinstance(current_plan,list) and len(current_plan)>0 and current_plan[0].startswith("Critical Error:")):
                    plan_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(current_plan))
                    cli_console.print(Markdown(f"## Execution Plan Proposed:\n{plan_md}"))
                elif current_plan and current_plan[0].startswith("Critical Error:"):
                    cli_console.print(Markdown(f"**PLANNING FAILED:** {current_plan[0]}"))
                    # The graph's decider should handle stopping.

            if "executor" in event:
                executed_step_idx = event["executor"].get("current_step_index", 1) -1
                full_plan = initial_state.get("plan", []) # Use the potentially updated plan
                plan_len = len(full_plan)

                step_instruction = "Unknown step"
                if 0 <= executed_step_idx < plan_len:
                    step_instruction = full_plan[executed_step_idx]

                cli_console.print(Markdown(f"### Executing Step {executed_step_idx + 1}/{plan_len}: *{step_instruction}*"))

                step_result = event["executor"]["step_results"][-1]
                cli_console.print(Markdown(f"**Result of Step {executed_step_idx + 1}:**\n```text\n{step_result}\n```"))

                if step_result.startswith("Critical Error:"):
                    cli_console.print(Markdown(f"**EXECUTION FAILED at step {executed_step_idx+1}.**"))
                    # Graph's decider should handle stopping.

        # After stream finishes, get the final accumulated state
        # This invoke might be redundant if the stream correctly processes all events and
        # the final state can be inferred, but it's safer for now.
        final_graph_state = buddy_app.invoke(initial_state, {"recursion_limit": 25})
        logging.debug(f"Raw final graph state from invoke: {final_graph_state}")

    except Exception as e:
        logging.error(f"Error during graph invocation: {e}", exc_info=True)
        cli_console.print(Markdown(f"\n**CRITICAL ERROR during graph execution:** {e}. Check logs."))
        # final_graph_state remains as it was before this exception or None if not set

    if final_graph_state:
        cli_console.print(Markdown("\n# --- Buddy AI Workflow Complete ---"))
        final_plan = final_graph_state.get('plan', [])
        if final_plan and isinstance(final_plan, list) and len(final_plan) > 0 and final_plan[0].startswith("Critical Error:"):
            cli_console.print(Markdown(f"**Workflow ended due to planner error:** {final_plan[0]}"))
        else:
            final_step_results = final_graph_state.get('step_results', [])
            if final_step_results:
                # Check if the last step result (which is a string) indicates critical error
                if isinstance(final_step_results[-1], str) and final_step_results[-1].startswith("Critical Error:"):
                     cli_console.print(Markdown(f"**Workflow ended due to executor error:** {final_step_results[-1]}"))
                else:
                    cli_console.print(Markdown("## --- Consolidated Output from All Steps ---"))
                    str_step_results = [str(res) for res in final_step_results]
                    final_consolidated_output = "\n\n---\n\n".join(str_step_results)
                    cli_console.print(Markdown(final_consolidated_output))
            else:
                cli_console.print(Markdown("*No step results, or workflow ended before execution.*"))
    else:
        cli_console.print(Markdown("\n**Graph execution failed or did not produce a final state.** See logs."))

    logging.info("Buddy application finished.")
    # Determine exit code based on success/failure
    # A successful run means final_graph_state is not None, and no critical errors in plan or last step result.
    successful_run = False
    if final_graph_state:
        plan_ok = not (final_graph_state.get('plan') and final_graph_state['plan'][0].startswith("Critical Error:"))
        results_ok = not (final_graph_state.get('step_results') and \
                          isinstance(final_graph_state['step_results'][-1], str) and \
                          final_graph_state['step_results'][-1].startswith("Critical Error:"))
        if plan_ok and results_ok:
            successful_run = True

    if successful_run:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
