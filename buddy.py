# buddy.py
import os
import argparse
import pathlib
from typing import TypedDict, List, Optional, Annotated
import operator
import traceback

from rich.console import Console
from rich.markdown import Markdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
import subprocess

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from typing import Type # Added for args_schema in WriteFileTool

from langgraph.prebuilt import create_react_agent

# Initialize Rich Console
console = Console()

DEFAULT_API_KEY = "AIzaSyAAfE6ydHeGx9-VVVVMbBLcMrB8QtGdpfE"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", DEFAULT_API_KEY)
if GOOGLE_API_KEY == DEFAULT_API_KEY:
    console.print(Markdown("Using **default GOOGLE_API_KEY** from script."))
else:
    console.print(Markdown("Using **GOOGLE_API_KEY** from environment."))
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

console.print(Markdown("# Buddy AI Agent Initializing..."))

def read_file_or_directory(path_str: str) -> str:
    path = pathlib.Path(path_str)
    content_parts = []
    # Using Markdown formatting for internal structure, though this function returns a string
    # that will be wrapped in a Markdown code block later if it's context.
    if path.is_file():
        try:
            content_parts.append(f"### Content from file: {path.name}\n```\n{path.read_text(encoding='utf-8', errors='ignore')}\n```")
        except Exception as e:
            console.print(Markdown(f"**Error reading file {path}:** {e}"))
            return ""
    elif path.is_dir():
        content_parts.append(f"### Content from directory: {path.name}\n")
        found_files_in_dir = False
        allowed_extensions = [
            ".txt", ".py", ".md", ".sh", ".json", ".yaml", ".yml", ".h", ".c", ".cc", ".cpp", ".java",
            ".js", ".ts", ".html", ".css", ".rb", ".php", ".pl", ".tcl", ".go", ".rs", ".swift",
            ".kt", ".scala", ".r", ".ps1", ".psm1", ".bat", ".cmd", ".vb", ".vbs", ".sql", ".xml",
            ".ini", ".cfg", ".conf", ".toml", ".dockerfile", "Dockerfile", ".tf"
        ]
        for item in path.rglob("*"): # Recursive glob
            if item.is_file() and (item.suffix.lower() in allowed_extensions or item.name == "Dockerfile"):
                try:
                    # Each file content will be in its own code block
                    content_parts.append(f"#### File: {item.relative_to(path)}\n```\n{item.read_text(encoding='utf-8', errors='ignore')}\n```")
                    found_files_in_dir = True
                except Exception as e:
                    console.print(Markdown(f"**Error reading file {item}:** {e}"))
        if not found_files_in_dir:
             content_parts.append("\nNo text files found in directory.")
    else:
        console.print(Markdown(f"**Error: Path '{path_str}' is not a valid file or directory.**"))
        return ""
    return "\n\n".join(content_parts)


def create_llm_instance(model_name_primary: str, model_name_fallback: str, llm_type: str):
    try:
        llm = ChatGoogleGenerativeAI(model=model_name_primary, temperature=0, convert_system_message_to_human=True)
        console.print(Markdown(f"**{llm_type} LLM created successfully using `{model_name_primary}`.**"))
        return llm
    except Exception as e:
        console.print(Markdown(f"**Error creating {llm_type} LLM with `{model_name_primary}`:** {e}. Trying fallback `{model_name_fallback}`."))
        try:
            llm = ChatGoogleGenerativeAI(model=model_name_fallback, temperature=0, convert_system_message_to_human=True)
            console.print(Markdown(f"**{llm_type} LLM created successfully using `{model_name_fallback}` (fallback).**"))
            return llm
        except Exception as e_fallback:
            console.print(Markdown(f"**Error creating {llm_type} LLM with fallback `{model_name_fallback}`:** {e_fallback}"))
            traceback.print_exc()
            return None

def create_executor_agent_runnable(llm):
    if not llm: return None
    shell_tool = ShellTool()
    python_script_tool = PythonScriptExecutorTool()
    write_file_tool = WriteFileTool() # Instantiate new tool
    tools = [shell_tool, python_script_tool, write_file_tool] # Add to tools list
    try:
        executor_agent = create_react_agent(llm, tools=tools)
        console.print(Markdown("**Executor ReAct agent created successfully with `ShellTool`, `PythonScriptExecutorTool`, and `WriteFileTool`.**"))
        return executor_agent
    except Exception as e:
        console.print(Markdown(f"**Error creating Executor Agent:** {e}"))
        traceback.print_exc()
        return None

class PythonScriptExecutorTool(BaseTool):
    name: str = "PythonScriptExecutorTool"
    description: str = "Executes a Python script and returns its output (stdout, stderr, return code). Input should be the path to the Python script."

    def _run(self, script_path: str) -> str:
        if not pathlib.Path(script_path).is_file():
            return f"Error: Script file does not exist at {script_path}"
        try:
            process = subprocess.run(
                ['python', script_path],
                capture_output=True, text=True, check=False, timeout=30
            )
            output_parts = [f"Executed '{script_path}'.", f"Exit Code: {process.returncode}"]
            if process.stdout:
                output_parts.append(f"Stdout:\n{process.stdout.strip()}")
            if process.stderr:
                output_parts.append(f"Stderr:\n{process.stderr.strip()}")
            return "\n".join(output_parts)
        except FileNotFoundError:
            return "Error: Python interpreter not found. Ensure Python is installed and in PATH."
        except subprocess.TimeoutExpired:
            return f"Error: Script execution timed out after 30 seconds ({script_path})."
        except Exception as e:
            return f"Error executing script '{script_path}': {str(e)}"

# Input model for WriteFileTool
class WriteFileToolInput(BaseModel):
    file_path: str = Field(description="The path to the file to write.")
    content: str = Field(description="The content to write to the file.")

# Tool for writing files
class WriteFileTool(BaseTool):
    name: str = "WriteFileTool"
    description: str = "Writes content to a specified file. Input should be the file path and the content."
    args_schema: Type[BaseModel] = WriteFileToolInput

    def _run(self, file_path: str, content: str) -> str:
        try:
            path = pathlib.Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists
            path.write_text(content, encoding='utf-8')
            return f"Successfully wrote content to {file_path}."
        except Exception as e:
            return f"Error writing to file '{file_path}': {str(e)}"

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
    "You are a master planning agent. Your goal is to create a detailed, step-by-step plan "
    "to achieve the user's objective. The user may provide context. "
    "Consider the context carefully. Each step in your plan MUST be a clear, atomic, actionable instruction "
    "for an execution agent. The agent has access to 'ShellTool' for general shell commands, "
    "'PythonScriptExecutorTool' for executing Python scripts, and 'WriteFileTool' for creating/writing files." # Updated tool info
    "If a step's purpose is to retrieve or display information (e.g., file content, command output), the instruction should explicitly include reporting or stating that information. "
    "Before performing an operation on a specific file (e.g., reading, writing, executing), if the file's existence or state is critical and might be ambiguous (e.g., it was just created or modified), consider adding a prior step to list files in its directory to confirm its status. "
    "To execute a Python script and capture its output, plan to use the 'PythonScriptExecutorTool' with the script's path as input. "
    "To write or create a file with specific content, plan to use the 'WriteFileTool' providing the 'file_path' and 'content'. " # Guidance for WriteFileTool
    "For other general shell commands, use 'ShellTool'. "
    "If the objective involves coding, break it down into: writing the code (using WriteFileTool), saving the code, " # Reinforced WriteFileTool
    "compiling (if necessary), and running/testing (using PythonScriptExecutorTool for .py files). "
    "If the objective involves analysis, break it down into steps to inspect files or run commands. "
    "User Objective: {objective}\n"
    "Context (if any):\n{context}\n" # Ensure context is handled if empty by .format
    "Respond ONLY with the structured plan."
)

def planner_node(state: BuddyGraphState) -> dict:
    """Generates a plan based on the objective and context."""
    console.print(Markdown("\n## --- Planner Node ---"))
    objective = state["objective"]
    context = state.get("context", "")

    global _planner_llm_structured
    if not _planner_llm_structured:
        console.print(Markdown("**CRITICAL ERROR: Planner LLM not initialized for planner_node.**"))
        return {
            "plan": ["Critical Error: Planner LLM not initialized."],
            "current_step_index": 0,
            "step_results": []
        }

    formatted_prompt = _PLANNER_PROMPT_TEMPLATE.format(objective=objective, context=context)
    console.print(f"Planner input (first 300 chars): {formatted_prompt[:300].strip()}...") # Keep this plain for now

    plan_steps = []
    try:
        ai_response = _planner_llm_structured.invoke(formatted_prompt)
        if ai_response and hasattr(ai_response, 'steps') and isinstance(ai_response.steps, list) and ai_response.steps:
            plan_steps = ai_response.steps
            if not all(isinstance(step, str) for step in plan_steps):
                console.print(Markdown(f"**Invalid plan structure (non-string steps):** {ai_response.steps}"))
                plan_steps = ["Critical Error: Planner returned non-string steps."]
        else:
            plan_steps = ["Critical Error: Planner returned no steps or invalid plan structure."]
            console.print(Markdown(f"**Invalid plan structure from LLM:** {ai_response}"))

        plan_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan_steps))
        console.print(Markdown(f"**Generated Plan:**\n{plan_md}"))

    except Exception as e:
        console.print(Markdown(f"**Error invoking structured planner LLM:** {e}"))
        traceback.print_exc()
        plan_steps = [f"Critical Error: Exception during planning - {str(e)}"]

    return {"plan": plan_steps, "current_step_index": 0, "step_results": []}


def executor_node(state: BuddyGraphState) -> dict:
    """Executes a single step of the plan using the ReAct agent."""
    console.print(Markdown("\n## --- Executor Node ---"))
    global _executor_agent_runnable_global
    if not _executor_agent_runnable_global:
        console.print(Markdown("**CRITICAL ERROR: Executor agent not initialized for executor_node.**"))
        return {
            "step_results": ["Critical Error: Executor agent not initialized."],
            "current_step_index": state.get("current_step_index", 0) + 1
        }

    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)

    step_output_str = "Error: Pre-execution state error in executor_node."
    next_step_idx = current_idx + 1

    if not plan or not isinstance(plan, list) or current_idx >= len(plan) or not plan[current_idx]:
        step_output_str = "Error: Invalid plan, step index out of bounds, or empty/invalid step."
        if plan and isinstance(plan, list) and len(plan) > 0 and plan[0].startswith("Critical Error"):
            step_output_str = plan[0]
    else:
        current_instruction = plan[current_idx]
        console.print(Markdown(f"**Executing step {current_idx + 1}/{len(plan)}:** {current_instruction}"))
        try:
            agent_input = {"messages": [HumanMessage(content=current_instruction)]}
            agent_response = _executor_agent_runnable_global.invoke(agent_input)

            if agent_response and "messages" in agent_response and agent_response["messages"]:
                step_output_str = str(agent_response["messages"][-1].content)
            else:
                step_output_str = "Error: No response or unexpected format from executor agent."
        except Exception as e:
            console.print(Markdown(f"**Error invoking executor agent for step '{current_instruction}':** {e}"))
            traceback.print_exc()
            step_output_str = f"Error executing step '{current_instruction}': {str(e)}"

    # For potentially long outputs, wrap in code block
    console.print(Markdown(f"**Step {current_idx + 1} Output:**\n```text\n{step_output_str}\n```"))
    return {"step_results": [step_output_str], "current_step_index": next_step_idx}


def should_continue_decider(state: BuddyGraphState) -> str:
    """Determines the next path in the graph (continue execution or end)."""
    console.print(Markdown("\n### --- Route Decider Node ---"))
    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)

    if not plan or not isinstance(plan, list) or len(plan) == 0 or not plan[0]:
        console.print(Markdown("*Decider: Invalid or empty plan. Ending.*"))
        return "end_workflow"

    if plan[0].startswith("Critical Error:"):
        console.print(Markdown("*Decider: Critical error in plan from planner node. Ending.*"))
        return "end_workflow"

    if current_idx >= len(plan):
        console.print(Markdown("*Decider: All steps executed or current index exceeds plan length. Ending.*"))
        return "end_workflow"

    console.print(Markdown(f"*Decider: Continuing to execute step {current_idx + 1}.*"))
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
            console.print(Markdown(f"**Prompt loaded from file:** `{args.prompt}`"))
        except Exception as e:
            console.print(Markdown(f"**Error reading prompt file `{args.prompt}`:** {e}"))
            exit(1)

    context_input_str = ""
    if args.context:
        console.print(Markdown(f"**Loading context from:** `{args.context}`"))
        context_input_str = read_file_or_directory(args.context) # This now returns a Markdown formatted string or empty

    console.print(Markdown(f"# User Objective\n\n{prompt_input}"))
    if context_input_str:
        console.print(Markdown(f"## Context Provided\n{context_input_str}")) # context_input_str is already MD formatted if from dir
    else:
        console.print(Markdown("--- \n*No context provided.*"))

    planner_llm_instance = create_llm_instance("gemini-1.5-flash-latest", "gemini-pro", "Planner")
    if not planner_llm_instance:
        exit("CRITICAL: Planner LLM could not be initialized. Exiting.")
    try:
        _planner_llm_structured = planner_llm_instance.with_structured_output(Plan)
        console.print(Markdown("**Planner LLM configured for structured output (Plan).**"))
    except Exception as e:
        console.print(Markdown(f"**CRITICAL: Failed to configure planner LLM for structured output:** {e}"))
        traceback.print_exc()
        exit(1)

    executor_llm_instance = create_llm_instance("gemini-1.5-flash-latest", "gemini-pro", "Executor")
    if not executor_llm_instance:
        exit("CRITICAL: Executor LLM could not be initialized. Exiting.")

    _executor_agent_runnable_global = create_executor_agent_runnable(executor_llm_instance)
    if not _executor_agent_runnable_global:
        exit("CRITICAL: Executor ReAct agent could not be created. Exiting.")

    workflow = StateGraph(BuddyGraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges("planner", should_continue_decider,
                                   {"continue_to_executor": "executor", "end_workflow": END})
    workflow.add_conditional_edges("executor", should_continue_decider,
                                   {"continue_to_executor": "executor", "end_workflow": END})

    app = workflow.compile()
    console.print(Markdown("\n**StateGraph compiled successfully.**"))

    initial_state = {
        "objective": prompt_input, "context": context_input_str, # Pass pre-formatted context string
        "plan": None, "current_step_index": 0,
        "step_results": [], "final_output": None
    }

    console.print(Markdown("\n# --- Invoking StateGraph ---"))
    final_graph_output_state = None
    try:
        # Simplified stream for brevity, focusing on invoke for final output display logic
        # for i, event_data in enumerate(app.stream(initial_state, {"recursion_limit": 25})):
        #     console.print(f"Stream Event {i+1}: {event_data}")
        final_graph_output_state = app.invoke(initial_state, {"recursion_limit": 25})
    except Exception as e:
        console.print(Markdown(f"\n**Error during graph invocation:** {e}"))
        traceback.print_exc()

    if final_graph_output_state:
        console.print(Markdown("\n# --- Graph Execution Complete ---"))
        # console.print(Markdown(f"**Objective:** {final_graph_output_state.get('objective')}")) # Already printed

        final_plan = final_graph_output_state.get('plan', [])
        plan_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(final_plan))
        console.print(Markdown(f"## Generated Plan:\n{plan_md if final_plan else '  No plan was generated or retained.'}"))

        final_step_results = final_graph_output_state.get('step_results', [])
        console.print(Markdown("## Execution Step Results:"))
        if final_step_results:
            for i, result_text in enumerate(final_step_results):
                console.print(Markdown(f"### Output of Step {i+1}:\n```text\n{result_text}\n```"))
        else:
            console.print(Markdown("*No step results recorded.*"))

        if final_step_results and not (final_plan and final_plan[0].startswith("Critical Error")):
             console.print(Markdown("## --- Consolidated Output (Joined Step Results) ---"))
             # Print each result as its own Markdown object to allow rich formatting if present
             for res_text in final_step_results:
                 console.print(Markdown(f"```text\n{res_text}\n```\n---"))
    else:
        console.print(Markdown("\n**Graph execution failed or did not produce a final state.**"))

    console.print(Markdown("\n**Buddy application finished.**"))
