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
    create_buddy_graph,
    set_agent_console # Added this
)

cli_console = Console()

def setup_logging():
    # It's good practice to ensure os is imported if using os.getenv
    # import os
    log_level_str = os.getenv("BUDDY_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level_str, logging.WARNING),
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging initialized with level {log_level_str}.")

def main():
    # cli_console is defined globally, but typically instantiated and used within main or other functions.
    # For clarity and ensuring it's the same instance, we can re-assign it here if needed,
    # or rely on the global one if its state is managed carefully.
    # However, the prompt asks to add the call *after* `cli_console = Console()`.
    # The existing code has `cli_console = Console()` at the global scope.
    # Let's assume the instruction meant to ensure it's set before use.
    # The global cli_console is already initialized when this module is imported.
    # We will call set_agent_console directly using the global cli_console.

    setup_logging()
    # Ensure agent console is set early
    set_agent_console(cli_console) # <--- Added this line
    logging.debug("Debug logging test message.")
    parser = argparse.ArgumentParser(description="Buddy AI Agent CLI - Your AI assistant for various tasks.")
    parser.add_argument("--prompt", type=str, required=True,
                        help="User objective, instructions, or a path to a text file containing them.")
    parser.add_argument("--context", type=str,
                        help="Path to a file or directory to provide as context to the agent.")
    parser.add_argument("--auto", action="store_true", help="Enable auto mode, bypassing human plan approval.")
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
        "plan": None,
        "current_step_index": 0,
        "step_results": [],
        "final_output": None,
        "user_feedback": None,
        "plan_approved": False,
        "auto_approve": args.auto
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

            if "human_approval" in event:
                # The human_approval_node itself will handle input if not auto_approve.
                # We just update the local state copy for clarity if needed,
                # and inform the user if interaction is expected.
                current_graph_state_from_event = event["human_approval"]
                initial_state.update(current_graph_state_from_event) # Keep our local state synced

                if not initial_state.get("auto_approve") and not initial_state.get("plan_approved") and not initial_state.get("user_feedback"):
                    cli_console.print(Markdown("--- \n*Waiting for human approval (Approve/Refine/Cancel)...*"))
                elif initial_state.get("auto_approve") and initial_state.get("plan_approved"):
                    cli_console.print(Markdown("--- \n*Plan auto-approved.*"))


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
        plan_approved = final_graph_state.get('plan_approved', False)
        user_feedback = final_graph_state.get('user_feedback')

        if final_plan and isinstance(final_plan, list) and len(final_plan) > 0 and final_plan[0].startswith("Critical Error:"):
            cli_console.print(Markdown(f"**Workflow ended due to PLANNER error:** {final_plan[0]}"))
        elif not plan_approved and not user_feedback and not final_graph_state.get("auto_approve"):
            # This condition implies cancellation before plan execution if not in auto_approve mode
            # and the plan itself wasn't a critical error from the start.
            # It also assumes 'END' was reached without plan_approved being True.
            if not (final_plan and isinstance(final_plan, list) and len(final_plan) > 0 and final_plan[0].startswith("Critical Error:")):
                 cli_console.print(Markdown("**Workflow ended: Plan was not approved and no feedback was provided (cancelled).**"))
            # If it was a critical error in the plan, that message is already shown or will be.
        else:
            final_step_results = final_graph_state.get('step_results', [])
            if final_step_results:
                if isinstance(final_step_results[-1], str) and final_step_results[-1].startswith("Critical Error:"):
                     cli_console.print(Markdown(f"**Workflow ended due to EXECUTOR error:** {final_step_results[-1]}"))
                elif not plan_approved and final_graph_state.get("auto_approve"):
                    # If auto_approve was on, but plan_approved is false, it implies an issue within human_approval_node or subsequent logic.
                    # This case might be redundant if human_approval_node handles its errors cleanly.
                    cli_console.print(Markdown("**Workflow ended: Plan was not auto-approved despite --auto flag. Check agent logic.**"))
                elif plan_approved: # Only show consolidated output if plan was approved and ran
                    cli_console.print(Markdown("## --- Consolidated Output from All Steps ---"))
                    str_step_results = [str(res) for res in final_step_results]
                    final_consolidated_output = "\n\n---\n\n".join(str_step_results)
                    cli_console.print(Markdown(final_consolidated_output))
                # If user_feedback is present, it implies it went to replan and then potentially ended.
                # The outcome of that replan (new plan, approval etc.) would be part of the loop.
                # If it ends with user_feedback still set, it's an unusual state, perhaps an error in graph logic.
                elif user_feedback:
                     cli_console.print(Markdown("**Workflow ended after user feedback was provided, but before further resolution. Check logs.**"))

            elif plan_approved: # Plan was approved but no steps were executed or no results.
                cli_console.print(Markdown("*Plan was approved, but no step results were recorded.*"))
            elif not plan_approved and not user_feedback and final_graph_state.get("auto_approve"):
                # Similar to above, if auto_approve was on but plan not approved.
                cli_console.print(Markdown("*Workflow ended: Plan was not auto-approved despite --auto flag and no steps executed. Check agent logic.*"))
            elif not plan_approved and not user_feedback : # General catch for other non-approved, non-feedback scenarios
                 pass # Already handled by the "Plan was not approved and no feedback was provided" message earlier.
            else:
                cli_console.print(Markdown("*No step results, or workflow ended before execution for other reasons.*"))
    else:
        cli_console.print(Markdown("\n**Graph execution failed critically or did not produce a final state.** See logs."))

    logging.info("Buddy application finished.")

    # Determine exit code
    successful_run = False
    if final_graph_state:
        plan_ok = not (final_graph_state.get('plan') and final_graph_state['plan'][0].startswith("Critical Error:"))
        # If plan was never approved (and not in auto_approve mode where approval is implicit for execution path)
        # then it's not a "successful run" in terms of execution.
        approved_for_execution = final_graph_state.get('plan_approved', False) or final_graph_state.get('auto_approve', False)

        last_step_result = final_graph_state.get('step_results', [])
        results_ok = not (last_step_result and \
                          isinstance(last_step_result[-1], str) and \
                          last_step_result[-1].startswith("Critical Error:"))

        # Objective met would be the ideal success, but for now, focus on error-free execution if approved
        # objective_met = final_graph_state.get("objective_met", False) # Assuming this might be added later

        if plan_ok and approved_for_execution and results_ok : # and objective_met (ideally)
            # If it was cancelled by user, it's not a "successful run" even if no technical errors
            if not (not final_graph_state.get("plan_approved") and \
                    not final_graph_state.get("user_feedback") and \
                    not final_graph_state.get("auto_approve")):
                successful_run = True

    if successful_run:
        cli_console.print(Markdown("--- \n**Workflow concluded successfully (no critical errors during execution).**"))
        sys.exit(0)
    else:
        cli_console.print(Markdown("--- \n**Workflow concluded with errors or was cancelled.**"))
        sys.exit(1)

if __name__ == "__main__":
    main()
