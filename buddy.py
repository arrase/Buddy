import os
import argparse
import pathlib
from typing import TypedDict, List, Optional, Annotated
import operator
import logging

from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Example: Control log level with environment variable
# log_level_str = os.getenv("BUDDY_LOG_LEVEL", "INFO").upper()
# logging.getLogger().setLevel(log_level_str)


# --- Rich Console Initialization ---
console = Console()

# --- API Key Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in environment. The application will likely fail.")
    # Let the program continue and fail at LLM instantiation, which will provide a clear error.
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Ensure it's set for langchain
    logging.info("GOOGLE_API_KEY found and set in environment.")

logging.info("Buddy AI Agent Initializing...")

def read_file_or_directory(path_str: str) -> str:
    path = pathlib.Path(path_str)
    content_parts = []
    if path.is_file():
        try:
            # For context, we want the raw content, not Markdown sub-headers within it
            file_content = path.read_text(encoding='utf-8', errors='ignore')
            content_parts.append(f"### Content from file: {path.name}\n```\n{file_content}\n```")
        except Exception as e:
            logging.error(f"Error reading file {path}: {e}")
            return f"Error reading file {path}: {e}" # Return error string for context
    elif path.is_dir():
        content_parts.append(f"### Content from directory: {path.name}\n")
        found_files_in_dir = False
        allowed_extensions = [
            ".txt", ".py", ".md", ".sh", ".json", ".yaml", ".yml", ".h", ".c", ".cc", ".cpp", ".java",
            ".js", ".ts", ".html", ".css", ".rb", ".php", ".pl", ".tcl", ".go", ".rs", ".swift",
            ".kt", ".scala", ".r", ".ps1", ".psm1", ".bat", ".cmd", ".vb", ".vbs", ".sql", ".xml",
            ".ini", ".cfg", ".conf", ".toml", ".dockerfile", "Dockerfile", ".tf"
        ]
        for item in path.rglob("*"):
            if item.is_file() and (item.suffix.lower() in allowed_extensions or item.name == "Dockerfile"):
                try:
                    file_content = item.read_text(encoding='utf-8', errors='ignore')
                    content_parts.append(f"#### File: {item.relative_to(path)}\n```\n{file_content}\n```")
                    found_files_in_dir = True
                except Exception as e:
                    logging.error(f"Error reading file {item} in directory {path_str}: {e}")
        if not found_files_in_dir:
             content_parts.append("\nNo text files found in directory.")
    else:
        logging.error(f"Path '{path_str}' is not a valid file or directory.")
        return f"Error: Path '{path_str}' is not a valid file or directory."
    return "\n\n".join(content_parts)


def create_llm_instance(model_name_primary: str, llm_type: str):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0)
        logging.info(f"{llm_type} LLM created successfully using gemini-1.5-flash-preview-04-17.")
        return llm
    except Exception as e:
        logging.error(f"Error creating {llm_type} LLM with gemini-1.5-flash-preview-04-17: {e}", exc_info=True)
        return None

def create_executor_agent_runnable(llm):
    if not llm: return None
    shell_tool = ShellTool()
    tools = [shell_tool]
    try:
        executor_agent = create_react_agent(llm, tools=tools)
        logging.info("Executor ReAct agent created successfully with ShellTool.")
        return executor_agent
    except Exception as e:
        logging.error(f"Error creating Executor Agent: {e}", exc_info=True)
        return None

class Plan(BaseModel):
    # No docstring here to avoid parsing issues
    steps: List[str] = Field(description="Actionable steps for the executor.") # Simplified description

class BuddyGraphState(TypedDict):
    objective: str
    context: Optional[str]
    plan: Optional[List[str]]
    current_step_index: int # Index of the *next* step to execute
    step_results: Annotated[List[str], operator.add] # Accumulates results from each step
    final_output: Optional[str] # Optional: For a final summarized result if needed

# Globals for initialized LLMs/Agents to be used by graph nodes
_planner_llm_structured: Optional[ChatGoogleGenerativeAI] = None
_executor_agent_runnable_global: Optional[callable] = None # ReAct agent is a runnable

# Planner prompt template
_PLANNER_PROMPT_TEMPLATE = (
    "You are a master planning agent. Your sole task is to create a detailed, step-by-step plan "
    "to achieve the user's stated objective. You have access ONLY to a `ShellTool` that can execute general shell commands. "
    "Therefore, every step in your plan that involves interaction with the system (creating files, running scripts, listing content, etc.) "
    "MUST be a precise, complete, and directly executable shell command string. "
    "The execution agent will pass this command string directly to the shell. "
    "User Objective: {objective}\n"
    "Context (if any):\n{context}\n\n"
    "Key Guidelines for Plan Steps:\n"
    "1.  **Shell Command Syntax:** Each action step must be a valid shell command. For example, to list files, a step would be: `ls -la`.\n"
    "2.  **Reporting Outputs:** If a step's purpose is to retrieve or display information (e.g., file content, command output, list of files), "
    "the plan must include an explicit instruction for the executor to report that information. E.g., `Execute 'cat file.txt' and report its content.`\n"
    "3.  **File Creation/Writing:**\n"
    "    *   To write content to a file, use `echo` or `printf`. For example: `echo 'This is line 1.' > /path/to/file.txt`\n"
    "    *   For multi-line content, `printf` is generally better: `printf 'First line.\\nSecond line.\\nThird line.' > /path/to/file.txt`\n"
    "    *   **Crucial for `echo/printf`:** Ensure the content string within the shell command is properly quoted, especially if it contains spaces, special shell characters (like $, !, `, quotes), or newlines. Single quotes (`'`) are generally safest for literal strings. If the content itself needs single quotes, you might need to use `echo \"content with 'single' quotes\" > file.txt` or more complex escaping.\n"
    "    *   To append to a file, use `>>` instead of `>`. E.g., `echo 'This appends.' >> /path/to/file.txt`\n"
    "    *   Before writing to a file in a new directory, ensure the directory exists: `mkdir -p /path/to/new_directory && echo 'content' > /path/to/new_directory/file.txt`\n"
    "4.  **Script Execution:**\n"
    "    *   To execute a Python script: `python /path/to/your_script.py`\n"
    "    *   To execute other shell scripts: `/path/to/your_script.sh` (ensure it's executable first with `chmod +x /path/to/your_script.sh`).\n"
    "    *   Always include a step to report the script's output.\n"
    "5.  **Verification:** After creating or modifying a file, consider adding a step to verify the change, e.g., `ls -l /path/to/file.txt` or `cat /path/to/file.txt` (and report its content).\n"
    "6.  **Atomicity:** Each step should ideally be a single, atomic shell command that achieves a specific part of the task.\n"
    "7.  **Quoting in Commands:** Pay close attention to shell quoting rules. If file paths or content have spaces or special characters, they MUST be quoted properly within the command string you generate. E.g., `echo 'content' > \"/path with spaces/file.txt\"`\n\n"
    "Respond ONLY with the structured plan (a list of strings, where each string is a natural language description of the step, which will be clear to the execution agent on how to formulate the shell command, or directly the shell command if it's simple enough). "
    "The execution agent is responsible for interpreting these steps and invoking ShellTool correctly based on your detailed, shell-focused instructions.\n"
    "Example of a good plan step: `Create the directory /tmp/my_app and then write the text 'Hello World' into a file named /tmp/my_app/greeting.txt. This involves using mkdir -p and then echo.`"
    "A more direct style for simple commands: `Execute the shell command: ls -l /tmp`"
    "Ensure the plan guides the executor to report relevant outputs after execution."
)

def planner_node(state: BuddyGraphState) -> dict:
    """Generates a plan based on the objective and context."""
    logging.info("Entering planner_node.")
    objective = state["objective"]
    context = state.get("context", "")

    global _planner_llm_structured
    # The _planner_llm_structured is initialized in __main__ and an error there exits the program.
    # If it's None here, it's an unexpected state, but the program would likely fail anyway.
    # The critical error check before was more for a state where it *could* be None due to some conditional logic.

    formatted_prompt = _PLANNER_PROMPT_TEMPLATE.format(objective=objective, context=context)
    logging.debug(f"Planner input prompt: {formatted_prompt}")

    plan_steps = []
    try:
        ai_response = _planner_llm_structured.invoke(formatted_prompt)
        if ai_response and hasattr(ai_response, 'steps') and isinstance(ai_response.steps, list) and ai_response.steps:
            plan_steps = ai_response.steps
            if not all(isinstance(step, str) for step in plan_steps):
                logging.error(f"Invalid plan structure (non-string steps): {ai_response.steps}")
                plan_steps = ["Critical Error: Planner returned non-string steps."]
        else:
            plan_steps = ["Critical Error: Planner returned no steps or invalid plan structure."]
            logging.error(f"Invalid plan structure from LLM: {ai_response}")

        logging.info(f"Generated plan: {plan_steps}")
        plan_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))
        console.print(Markdown(f"## Execution Plan Proposed:\n{plan_md}"))

    except Exception as e:
        logging.error(f"Error invoking structured planner LLM: {e}", exc_info=True)
        plan_steps = [f"Critical Error: Exception during planning - {str(e)}"]

    return {"plan": plan_steps, "current_step_index": 0, "step_results": []}


def executor_node(state: BuddyGraphState) -> dict:
    """Executes a single step of the plan using the ReAct agent."""
    logging.info(f"Entering executor_node for step index {state.get('current_step_index', 0)}.")
    global _executor_agent_runnable_global
    # _executor_agent_runnable_global is initialized in __main__ and an error there exits the program.

    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)

    step_output_str = "Error: Pre-execution state error in executor_node." # Default error
    next_step_idx = current_idx + 1

    if not plan or not isinstance(plan, list) or not (0 <= current_idx < len(plan)) or not plan[current_idx]:
        step_output_str = "Error: Invalid plan, step index out of bounds, or empty step content."
        logging.error(step_output_str)
        # Propagate critical error from planner if it's the source
        if plan and isinstance(plan, list) and len(plan) > 0 and plan[0].startswith("Critical Error"):
            step_output_str = plan[0]
    else:
        current_instruction = plan[current_idx]
        console.print(Markdown(f"### Executing Step {current_idx + 1}/{len(plan)}: *{current_instruction}*"))
        logging.info(f"Executing step {current_idx + 1}/{len(plan)}: {current_instruction}")
        try:
            agent_input = {"messages": [HumanMessage(content=current_instruction)]}
            # The agent's response is a dict with "messages", last one is AI's output
            agent_response = _executor_agent_runnable_global.invoke(agent_input)
            logging.debug(f"Raw agent response for step {current_idx + 1}: {agent_response}")

            if agent_response and "messages" in agent_response and agent_response["messages"]:
                step_output_str = str(agent_response["messages"][-1].content)
            else:
                step_output_str = "Error: No response or unexpected format from executor agent."
                logging.error(f"Unexpected agent response format for step {current_idx + 1}: {agent_response}")
        except Exception as e:
            logging.error(f"Error invoking executor agent for step '{current_instruction}': {e}", exc_info=True)
            step_output_str = f"Error executing step '{current_instruction}': {str(e)}"

    console.print(Markdown(f"**Result of Step {current_idx + 1}:**\n```text\n{step_output_str}\n```"))
    logging.info(f"Step {current_idx + 1} output: {step_output_str}")
    return {"step_results": [step_output_str], "current_step_index": next_step_idx}


def should_continue_decider(state: BuddyGraphState) -> str:
    """Determines the next path in the graph (continue execution or end)."""
    logging.info("Entering should_continue_decider.")
    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)

    # Simplified check for plan validity and completion
    if not plan or not isinstance(plan, list) or current_idx >= len(plan):
        if current_idx >= len(plan) and plan and len(plan) > 0 : # Check if plan exists and has items
             logging.info("Decider: All steps executed or current index exceeds plan length. Ending workflow.")
        else:
             logging.info("Decider: Invalid or empty plan. Ending workflow.")
        return "end_workflow"

    # Check for critical error propagated from planner
    if plan[0].startswith("Critical Error:"):
        logging.info("Decider: Critical error in plan from planner node. Ending workflow.")
        return "end_workflow"

    logging.info(f"Decider: Continuing to execute step {current_idx + 1}.")
    return "continue_to_executor"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buddy AI Agent CLI")
    parser.add_argument("--prompt", type=str, required=True, help="User objective or instructions (string or filepath)")
    parser.add_argument("--context", type=str, help="Path to a file or directory for context")
    args = parser.parse_args()

    prompt_input = args.prompt
    if pathlib.Path(prompt_input).is_file():
        try:
            prompt_input = pathlib.Path(prompt_input).read_text(encoding='utf-8')
            logging.info(f"Prompt loaded from file: {args.prompt}")
        except Exception as e:
            logging.error(f"Error reading prompt file {args.prompt}: {e}", exc_info=True)
            console.print(Markdown(f"**Critical Error:** Could not read prompt file `{args.prompt}`. See logs for details."))
            exit(1)

    context_input_str = ""
    if args.context:
        logging.info(f"Loading context from: {args.context}")
        context_input_str = read_file_or_directory(args.context)

    console.print(Markdown(f"# User Objective\n\n{prompt_input}"))
    if context_input_str:
        # Assuming context_input_str is already Markdown formatted by read_file_or_directory
        console.print(Markdown(f"## Context Provided\n{context_input_str}"))
    else:
        console.print(Markdown("--- \n*No context provided.*"))

    logging.info("Initializing LLMs and Agent...")
    planner_llm_instance = create_llm_instance("gemini-2.5-flash-preview-04-17", "Planner")
    if not planner_llm_instance:
        console.print(Markdown("**CRITICAL ERROR:** Planner LLM failed to initialize. Buddy cannot proceed."))
        exit("CRITICAL: Planner LLM could not be initialized. Exiting.")
    try:
        _planner_llm_structured = planner_llm_instance.with_structured_output(Plan)
        logging.info("Planner LLM configured for structured output (Plan).")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to configure planner LLM for structured output: {e}", exc_info=True)
        console.print(Markdown("**CRITICAL ERROR:** Failed to configure planner. Buddy cannot proceed."))
        exit(1)

    executor_llm_instance = create_llm_instance("gemini-2.5-flash-preview-04-17", "Executor")
    if not executor_llm_instance:
        console.print(Markdown("**CRITICAL ERROR:** Executor LLM failed to initialize. Buddy cannot proceed."))
        exit("CRITICAL: Executor LLM could not be initialized. Exiting.")

    _executor_agent_runnable_global = create_executor_agent_runnable(executor_llm_instance)
    if not _executor_agent_runnable_global:
        console.print(Markdown("**CRITICAL ERROR:** Executor ReAct agent failed to create. Buddy cannot proceed."))
        exit("CRITICAL: Executor ReAct agent could not be created. Exiting.")

    logging.info("Defining StateGraph workflow...")
    workflow = StateGraph(BuddyGraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", should_continue_decider,
                                   {"continue_to_executor": "executor", "end_workflow": END})
    workflow.add_conditional_edges("executor", should_continue_decider,
                                   {"continue_to_executor": "executor", "end_workflow": END})

    app = workflow.compile()
    logging.info("StateGraph compiled successfully.")

    initial_state = {
        "objective": prompt_input, "context": context_input_str,
        "plan": None, "current_step_index": 0,
        "step_results": [], "final_output": None
    }

    console.print(Markdown("\n# --- Buddy AI Workflow Starting ---"))
    logging.info("Invoking StateGraph workflow.")
    final_graph_output_state = None
    try:
        final_graph_output_state = app.invoke(initial_state, {"recursion_limit": 25})
        logging.debug(f"Raw final graph state: {final_graph_output_state}")
    except Exception as e:
        logging.error(f"Error during graph invocation: {e}", exc_info=True)
        console.print(Markdown(f"\n**CRITICAL ERROR during graph execution:** {e}. Check logs for details."))

    if final_graph_output_state:
        console.print(Markdown("\n# --- Buddy AI Workflow Complete ---"))

        final_plan = final_graph_output_state.get('plan', [])
        final_step_results = final_graph_output_state.get('step_results', [])

        if final_step_results and not (final_plan and len(final_plan) > 0 and final_plan[0].startswith("Critical Error")):
             console.print(Markdown("## --- Consolidated Output ---"))
             final_consolidated_output = "\n\n---\n\n".join(map(str,final_step_results))
             console.print(Markdown(final_consolidated_output))
        elif final_plan and len(final_plan) > 0 and final_plan[0].startswith("Critical Error"):
            console.print(Markdown(f"**Workflow ended due to planner error:** {final_plan[0]}"))
        else:
            console.print(Markdown("*No step results to consolidate, or workflow ended prematurely.*"))
    else:
        console.print(Markdown("\n**Graph execution failed or did not produce a final state.** See logs for details."))

    logging.info("Buddy application finished.")
