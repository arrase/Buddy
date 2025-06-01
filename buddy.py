import os # Already imported, but ensure it's used for getcwd
import json
import argparse
import sys # Added for sys.exit
import subprocess # Added for shell tool
from langchain.tools import Tool # Added for shell tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from rich.console import Console # Added for Rich output
from rich.markdown import Markdown # Added for Rich output
from rich.panel import Panel # Added for Rich security warning

# 1. Imports and Setup
GOOGLE_API_KEY = "AIzaSyDtdc1YKLn3INvgrHX_LOIKXz1SRf36irU"  # Replace with your actual key if necessary
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 2. Define Agent State
class PlanExecuteState(TypedDict):
    prompt: str
    plan: List[str]
    past_steps: List[tuple[str, str]]
    context: str
    next_step_index: int

# 3. Initialize LLM
# Global LLM variable, will be initialized in main after API key check
llm = None

# Rich Console
console = Console()

# Agent's Current Working Directory
AGENT_WORKING_DIRECTORY = os.getcwd()

# Shell Command Execution Tool
def execute_shell_command(command: str) -> str:
    global AGENT_WORKING_DIRECTORY # Declare intention to modify global variable
    """
    Executes a shell command, managing a virtual working directory, and returns its output.
    Handles 'cd' commands by changing the virtual working directory.
    """
    try:
        command_trimmed = command.strip()

        # Handle 'cd' command
        if command_trimmed.startswith("cd "):
            path_to_change = command_trimmed[3:].strip()
            if not path_to_change or path_to_change == "~" or path_to_change == "$HOME":
                # Basic handling for 'cd' to home. More robust handling might use os.path.expanduser('~')
                # For simplicity, we'll restrict 'cd' to explicit paths or prevent 'cd' without args.
                # For now, let's require an explicit path.
                user_home = os.path.expanduser('~')
                if command_trimmed == "cd" or command_trimmed == "cd ~" or command_trimmed == f"cd {user_home}":
                     AGENT_WORKING_DIRECTORY = user_home
                     success_msg = f"Changed virtual working directory to: {AGENT_WORKING_DIRECTORY}"
                     print(success_msg)
                     return success_msg
                elif not path_to_change:
                    return "Error: 'cd' without a specific path argument is not fully supported. Use 'cd ~' or 'cd /path/to/dir'."


            # Resolve new path
            if os.path.isabs(path_to_change):
                new_dir = path_to_change
            else:
                new_dir = os.path.join(AGENT_WORKING_DIRECTORY, path_to_change)

            new_dir = os.path.normpath(new_dir) # Normalize path (e.g., ..)

            if os.path.isdir(new_dir):
                AGENT_WORKING_DIRECTORY = new_dir
                success_msg = f":heavy_check_mark: Changed virtual working directory to: `{AGENT_WORKING_DIRECTORY}`"
                console.print(Markdown(success_msg)) # Internal print
                return success_msg # Return this for execute_step to render
            else:
                error_msg = f":x: Error changing directory: `{new_dir}` is not a valid directory (from CWD: `{AGENT_WORKING_DIRECTORY}`)."
                console.print(Markdown(error_msg)) # Internal print
                return error_msg # Return this

        # For other commands, execute in AGENT_WORKING_DIRECTORY
        # Internal print before execution:
        console.print(f"Executing: `{command}` in CWD: `{AGENT_WORKING_DIRECTORY}`")

        process = subprocess.run(
            command,
            shell=True, # Still using shell=True for convenience; be mindful of security.
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=AGENT_WORKING_DIRECTORY # Use the agent's current working directory
        )

        # Construct the return string with Markdown formatting
        output_parts = []
        stdout_content = process.stdout.strip()
        stderr_content = process.stderr.strip()

        if stdout_content:
            output_parts.append(f"**Stdout:**\n```text\n{stdout_content}\n```")
        if stderr_content:
            output_parts.append(f"**Stderr:**\n```text\n{stderr_content}\n```")

        if not output_parts and process.returncode == 0:
            output_parts.append("*Command executed successfully with no output.*")

        # Add return code information if it's non-zero, or if it succeeded with no textual output (already covered)
        # or if there was stderr even on success.
        if process.returncode != 0:
            output_parts.append(f"**Return Code:** `{process.returncode}`")
        elif not output_parts and process.returncode !=0 : # Should be covered by the one above.
             output_parts.append(f"*Command failed with return code {process.returncode} and no output.*")

        returned_output_string = "\n\n".join(output_parts).strip()

        # Internal log of what's being returned (optional, can be verbose)
        # console.print(Markdown(f"*Shell tool returning to executor:*\n{returned_output_string}"))

        return returned_output_string

    except subprocess.TimeoutExpired:
        error_message_md = f":x: **Timeout Error:** Command `{command}` timed out after 30 seconds (CWD: `{AGENT_WORKING_DIRECTORY}`)."
        console.print(Markdown(error_message_md)) # Internal print
        return error_message_md # Return this for execute_step
    except Exception as e:
        error_message_md = f":x: **Execution Error:** While running `{command}` (CWD: `{AGENT_WORKING_DIRECTORY}`): `{e}`"
        console.print(Markdown(error_message_md)) # Internal print
        return error_message_md # Return this

shell_tool = Tool(
    name="ShellCommandExecutor",
    func=execute_shell_command,
    description="Executes a given shell command and returns its standard output and standard error. Use this for tasks requiring interaction with the operating system, like file manipulation, package management, or running scripts."
)

# 4. Implement Planner Node
def parse_llm_list_output(text: str) -> List[str]:
    """Parses LLM text output that should be a list into a Python list."""
    # Remove potential markdown list characters and surrounding quotes
    lines = text.strip().split('\n')
    parsed_list = []
    for line in lines:
        # Remove leading/trailing whitespace and list markers (e.g., "1. ", "- ")
        cleaned_line = line.strip()
        if cleaned_line.startswith(("* ", "- ")):
            cleaned_line = cleaned_line[2:]
        elif cleaned_line and cleaned_line[0].isdigit() and cleaned_line[1:3] in (". ", "."):
            cleaned_line = cleaned_line.split(". ", 1)[-1]

        # Remove surrounding quotes if any
        if cleaned_line.startswith('"') and cleaned_line.endswith('"'):
            cleaned_line = cleaned_line[1:-1]
        if cleaned_line.startswith("'") and cleaned_line.endswith("'"):
            cleaned_line = cleaned_line[1:-1]

        if cleaned_line: # Avoid adding empty strings
            parsed_list.append(cleaned_line)
    return parsed_list

PLANNER_PROMPT_TEMPLATE = """You are a helpful AI assistant. Your goal is to create a step-by-step plan to accomplish the user's request.
The user may ask for tasks related to general knowledge, programming, or Linux system administration.

You have an executor that can perform the following actions:
1.  Answer questions or generate text based on its knowledge and the provided context.
2.  Execute shell commands on a Linux system using a 'ShellCommandExecutor' tool. This tool can run commands like ls, cat, python, pip, apt, etc.

When creating the plan, if a step requires running a shell command, clearly state the command to be run.
For example:
- "Run the command `ls -la` to list directory contents."
- "Install the 'requests' Python library using the command `pip install requests`."
- "Check the Python version with the command `python --version`."

Break down the user's request into a series of clear, numbered steps.
The output should be a list of strings, where each string is a step in the plan.

User Request: {prompt}
Context (if any): {context}

Respond *only* with the numbered list of plan steps. Do not add any other text, preamble, or explanation.
Example format:
1. First step.
2. Second step, which might involve a command like `echo "hello"`.
3. Third step.
"""
PLANNER_PROMPT = ChatPromptTemplate.from_template(PLANNER_PROMPT_TEMPLATE)

def get_planner(llm):
    planner_chain = PLANNER_PROMPT | llm
    # The content extraction and parsing will be handled in plan_step
    return planner_chain

# 5. Implement Executor Node
EXECUTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful AI assistant that executes a single step from a plan. "
         "You are given the overall 'prompt' (overall goal), the 'context', the 'plan' (list of all steps), "
         "the 'past_steps' (already executed steps and their results), and the 'current_step' to execute. "
         "Focus *only* on the 'current_step'. Do not try to execute other steps. "
         "\n\nYou have access to the following tool:\n"
         f"- {shell_tool.name}: {shell_tool.description}\n\n"
         "If you need to use the ShellCommandExecutor tool to accomplish the current step, respond *only* with a JSON object in the following format:\n"
         '```json\n'
         '{{\n'
         '  "tool_to_use": "ShellCommandExecutor",\n'
         '  "command_to_run": "your shell command here"\n'
         '}}\n'
         '```\n'
         "Ensure the JSON is perfectly formatted. Do not add any text before or after the JSON object if using the tool.\n"
         "If you do not need to use the tool, provide your direct answer or result for completing the step as a plain string. "
         "If the step is to write code, provide only the code. "
         "If the step is to answer a question, provide only the answer."),
        ("human",
         "Overall Prompt (Goal): {prompt}\n"
         "Context: {context}\n"
         "Full Plan:\n{plan_str}\n"
         "Past Steps (step, result):\n{past_steps_str}\n"
         "Current Step to Execute: {current_step}")
    ]
)

def get_executor(llm):
    executor_chain = EXECUTOR_PROMPT | llm
    # We will get raw AIMessage and extract content in execute_step, as we need to check for tool use first.
    return executor_chain

# 6. Define Graph Logic Functions
def execute_step(state: PlanExecuteState):
    executor_chain = get_executor(llm) # llm should be initialized by now
    current_step_description = state['plan'][state['next_step_index']]

    plan_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state['plan']))
    # Format past_steps for the prompt
    past_steps_formatted = "\n".join([f"Step: {ps[0]}\nResult: {ps[1]}" for ps in state['past_steps']]) \
        if state['past_steps'] else "No steps executed yet."

    step_number = state['next_step_index'] + 1
    # Use a rule to separate steps, including a truncated step description for context
    rule_title = f"[bold cyan]Executing Step {step_number}: {current_step_description[:70]}{'...' if len(current_step_description) > 70 else ''}"
    console.rule(rule_title)
    # Optionally, print the full action if needed, but it's in the rule.
    # console.print(Markdown(f"**Full Action:** {current_step_description}"))


    step_output: str
    try:
        # Get raw AIMessage from LLM
        raw_executor_response_message = executor_chain.invoke({
            "prompt": state['prompt'],
            "context": state['context'],
            "plan_str": plan_str,
            "past_steps_str": past_steps_formatted, # Use the formatted string
            "current_step": current_step_description
        })

        # Extract content from AIMessage
        raw_executor_response = raw_executor_response_message.content if hasattr(raw_executor_response_message, 'content') else str(raw_executor_response_message)

        # Attempt to parse for tool use
        try:
            # Clean the response: remove markdown, leading/trailing whitespace.
            cleaned_response = raw_executor_response.strip()
            if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("`json") and cleaned_response.endswith("`"): # Handle single backtick
                 cleaned_response = cleaned_response[5:-1].strip()
            elif cleaned_response.startswith("{") and cleaned_response.endswith("}"): # Already looks like JSON
                pass # Use as is
            # else: It's not wrapped in JSON markdown, assume it's a direct string or malformed.

            # Try to parse the cleaned string as JSON.
            tool_call_request = json.loads(cleaned_response)

            # Check if it's a dictionary and if it's a request for our shell tool.
            if isinstance(tool_call_request, dict) and \
               tool_call_request.get("tool_to_use") == shell_tool.name:
                command = tool_call_request.get("command_to_run")

                # Validate the command.
                if command and isinstance(command, str):
                    console.print(Markdown(f"Executing shell command: `{command}`"))
                    step_output = shell_tool.run(command) # Execute the command.
                elif command: # Command is present but not a string.
                     error_msg = f"Error: {shell_tool.name} was called, but 'command_to_run' was not a valid string: Got '{command}' (type: {type(command).__name__})."
                     console.print(Markdown(f"**Shell Tool Error:** {error_msg}"))
                     step_output = error_msg
                else: # Command is missing.
                    error_msg = f"Error: {shell_tool.name} was called, but no 'command_to_run' was provided."
                    console.print(Markdown(f"**Shell Tool Error:** {error_msg}"))
                    step_output = error_msg
            else:
                # It was valid JSON, but not a request for the ShellCommandExecutor tool.
                # Treat as a direct answer from the LLM.
                step_output = raw_executor_response
        except json.JSONDecodeError:
            # The response was not valid JSON. Treat as a direct answer from the LLM.
            step_output = raw_executor_response
        except Exception as e:
            # Catch any other errors during the parsing or tool logic.
            error_text = f"Error processing executor response or during tool call attempt: {e}"
            console.print(Markdown(f"**Execution Error:** {error_text}"))
            step_output = f"Error during step execution logic: {e}"

    except Exception as e: # Catch errors from the LLM call itself.
        error_text = f"Error during executor LLM call for step '{current_step_description}': {e}"
        console.print(Markdown(f"**LLM Call Error:** {error_text}"))
        step_output = f"Error executing step (LLM call failed): {e}"

    console.print(Markdown("**Result:**"))
    console.print(Markdown(step_output if step_output.strip() else "*No output from step.*"))
    # console.print("--------------------------") # Replaced by console.rule at the start of next step or final summary
    console.print("") # Add a blank line for spacing

    updated_past_steps = state['past_steps'] + [(current_step_description, step_output)]
    next_index = state['next_step_index'] + 1

    return {"past_steps": updated_past_steps, "next_step_index": next_index}

def plan_step(state: PlanExecuteState):
    planner_chain = get_planner(llm) # llm should be initialized
    generated_plan: List[str] = []

    try:
        response = planner_chain.invoke({"prompt": state["prompt"], "context": state["context"]})

        plan_str = ""
        if hasattr(response, 'content'):
            plan_str = response.content
        elif isinstance(response, str): # Should not happen with LLM but good for robustness
            plan_str = response
        else:
            print(f"Warning: Planner output was not a string or AIMessage with content: {type(response)}")
            generated_plan = ["Planner failed to produce a valid plan string format."]
            # Early return or set plan to indicate failure
            return {"plan": generated_plan, "next_step_index": 0, "past_steps": []}

        # Use existing parse_llm_list_output function
        generated_plan = parse_llm_list_output(plan_str)

        if not generated_plan: # If parsing resulted in an empty plan
            print("Warning: Planner produced an empty plan or parsing failed.")
            # Provide a default or error plan
            generated_plan = ["No actionable steps identified by the planner or plan was empty."]
            console.print(Markdown("*Planner Warning: No actionable steps identified or plan was empty.*"))


    except Exception as e:
        error_msg = f"Error during planner LLM call or plan parsing: {e}"
        console.print(Markdown(f"# Plan Generation Failed\n\n**Error:** {error_msg}"))
        # Return a state that indicates failure
        generated_plan = [f"Failed to generate a plan due to an error: {e}"]
        return {"plan": generated_plan, "next_step_index": 0, "past_steps": []}

    console.rule("[bold cyan]Execution Plan")
    if generated_plan and not (len(generated_plan) == 1 and "Failed to generate a plan" in generated_plan[0]):
        # Prepare the plan as a numbered list in a single Markdown string
        markdown_plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(generated_plan)])
        console.print(Markdown(markdown_plan_str))
    elif not generated_plan: # Should be caught by earlier logic, but as a safeguard
        console.print(Markdown("*No plan steps were generated.*"))
    # If plan generation failed, the error is already printed above.
    # No need to print "Failed to generate plan..." again unless it's the only content of generated_plan
    elif len(generated_plan) == 1 and "Failed to generate a plan" in generated_plan[0] and "Error:" not in generated_plan[0]:
         # This case is when the generated_plan literally is ["Failed to generate a plan due to an error: {e}"]
         # but the detailed error was already printed by the except block.
         # So we just print the plan title and the single error message if it wasn't the detailed one.
         # console.print(Markdown(generated_plan[0])) # Already printed by the except block more clearly
         pass


    console.print("") # Add a blank line for spacing after the plan

    # Initialize past_steps here to ensure it's always present in the state
    # even if planning is the first step after a restart/retry.
    return {"plan": generated_plan, "next_step_index": 0, "past_steps": []}

def should_continue(state: PlanExecuteState):
    if state['next_step_index'] < len(state['plan']):
        return "execute"
    return "end"

# 7. Create the Graph
workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", plan_step)
workflow.add_node("executor", execute_step)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "execute": "executor",
        "end": END,
    },
)

app = workflow.compile()

# Helper Functions for Input Handling
def read_file_content(filepath: str) -> str:
    """Reads and returns the content of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        console.print(Markdown(f":x: [bold red]File Read Error:[/bold red] File not found at `{filepath}`"))
        sys.exit(1)
    except IOError as e:
        console.print(Markdown(f":x: [bold red]File Read Error:[/bold red] Error reading file `{filepath}`: {e}"))
        sys.exit(1)

def read_directory_content(dir_path: str) -> str:
    """Recursively reads content of all files in a directory."""
    if not os.path.isdir(dir_path):
        console.print(Markdown(f":x: [bold red]Directory Read Error:[/bold red] Directory not found at `{dir_path}`"))
        sys.exit(1)

    console.print(Markdown(f":mag: Reading content from directory: `{dir_path}`"), style="dim")
    all_content = []
    for root, _, files_in_dir in os.walk(dir_path): # Renamed 'files' to 'files_in_dir'
        for file_name in files_in_dir: # Renamed 'file' to 'file_name'
            filepath = os.path.join(root, file_name)
            try:
                console.print(Markdown(f"...reading `{filepath}`"), style="dim")
                file_content = read_file_content(filepath)
                # Format context for better readability when presented to LLM or in logs
                all_content.append(f"\n**Source: `{filepath}`**\n```text\n{file_content}\n```")
            except SystemExit: # Propagate SystemExit from read_file_content
                raise
            except Exception as e:
                console.print(Markdown(f":warning: Skipping file `{filepath}` due to error: {e}"), style="yellow")
    return "\n".join(all_content)

# 8. Main Execution Logic
if __name__ == "__main__":
    # Security Warning
    warning_title = "[bold red]!!! WARNING: Buddy AI Agent - Shell Command Execution !!![/bold red]"
    warning_message = (
        "This agent can execute shell commands generated by an AI model.\n"
        "Executing AI-generated commands can be [bold red]DANGEROUS[/bold red] and may lead to:\n"
        "- Unintended system modifications\n"
        "- Data loss or corruption\n"
        "- Security vulnerabilities\n\n"
        "ALWAYS review the generated plan and the specific commands before execution if possible, "
        "and only run this agent in a safe, isolated environment if you are unsure.\n"
        "[italic]You are responsible for any actions taken by this agent.[/italic]"
    )
    console.print(Panel(Markdown(warning_message), title=warning_title, border_style="red", expand=False))
    console.print("") # For spacing

    # API Key Check
    if not GOOGLE_API_KEY:
        console.print(Markdown(":x: [bold red]Error:[/bold red] `GOOGLE_API_KEY` is not set. Please set it in the script or as an environment variable."), style="red")
        sys.exit(1)

    # Initialize LLM after key check
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
        console.print(Markdown(f":robot: LLM initialized with `gemini-2.5-flash-preview-04-17`."), style="dim")
    except Exception as e:
        console.print(Markdown(f":warning: Error initializing LLM with `gemini-2.5-flash-preview-04-17`: {e}. Attempting fallback..."), style="yellow")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
            console.print(Markdown(f":robot: LLM initialized with fallback `gemini-pro`."), style="dim")
        except Exception as e_pro:
            console.print(Markdown(f":x: [bold red]LLM Initialization Failed:[/bold red] Error with `gemini-pro` as well: {e_pro}. Please check API key and model availability."), style="red")
            sys.exit(1)

    parser = argparse.ArgumentParser(description="Buddy Agent: A helpful AI assistant, enhanced with Rich output.")
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="User prompt or path to a file containing the prompt."
    )
    parser.add_argument(
        "-c", "--context",
        required=False,
        default="",
        help="Path to a file or directory to load context from."
    )
    args = parser.parse_args()

    # Process Prompt
    prompt_input = args.prompt
    user_prompt: str
    try:
        if os.path.isfile(prompt_input):
            console.print(Markdown(f":page_facing_up: Reading prompt from file: `{prompt_input}`"), style="dim")
            user_prompt = read_file_content(prompt_input)
        else:
            user_prompt = prompt_input # Use input directly as prompt
    except Exception as e: # read_file_content now exits on error, this is a fallback
        console.print(Markdown(f":x: [bold red]Failed to read prompt:[/bold red] {e}"), style="red")
        exit(1)

    # Process Context
    context_input = args.context
    context_data = ""
    if context_input:
        # console.print(Markdown(f":file_folder: Loading context from: `{context_input}`"), style="dim") # Done by read_directory_content
        try:
            if os.path.isfile(context_input):
                context_data = read_file_content(context_input)
            elif os.path.isdir(context_input):
                context_data = read_directory_content(context_input) # This now prints its own status
            else:
                console.print(Markdown(f":warning: Context path `{context_input}` is not a valid file or directory. Proceeding without context."), style="yellow")
        except Exception as e: # read_file_content/read_directory_content exit on error, this is a fallback
            console.print(Markdown(f":x: [bold red]Failed to load context from `{context_input}`:[/bold red] {e}. Proceeding without context."), style="red")
            context_data = ""

    # Prepare Initial State for LangGraph
    initial_state = {
        "prompt": user_prompt,
        "context": context_data,
    }

    # User feedback
    console.rule("[bold blue]Buddy Agent Initializing")
    console.print(Markdown(f"**User Prompt:**\n```text\n{user_prompt}\n```"))
    if context_data:
        # Display only a preview if context is very long
        context_display_limit = 1000
        if len(context_data) > context_display_limit:
             console.print(Markdown(f"**Context Provided (Preview):**\n```text\n{context_data[:context_display_limit]}...\n```" ))
        else:
             console.print(Markdown(f"**Context Provided:**\n```text\n{context_data}\n```" ))
    else:
        console.print(Markdown("*No context provided.*"))
    console.print("") #Spacing

    # Invoke the agent. The plan and step-by-step execution will be printed by the nodes.
    final_state = {} # Initialize to ensure it's available for summary
    try:
        # The individual nodes (plan_step, execute_step) now handle their own Rich printing for live updates.
        final_state = app.invoke(initial_state)
    except Exception as e:
        console.print(Markdown(f"\n:x: [bold red]Agent Execution Failed Critically:[/bold red]\nAn unexpected error occurred during the main agent execution loop: {e}"))
        # final_state will remain as it was at the point of error or empty if error was at start.

    console.rule("[bold green]Agent Execution Complete")

    console.print(Markdown("### Final Plan Executed:"))
    if 'plan' in final_state and final_state.get('plan'):
        md_final_plan = "\n".join([f"{i+1}. {step}" for i, step in enumerate(final_state['plan'])])
        console.print(Markdown(md_final_plan if md_final_plan.strip() else "*Plan was empty or contained no actionable steps.*"))
    else:
        console.print(Markdown("*No plan was part of the final state or it was empty.*"))
    console.print("")

    console.print(Markdown("### Execution History:"))
    if 'past_steps' in final_state and final_state.get('past_steps'):
        if not final_state['past_steps']:
            console.print(Markdown("*No steps were executed or recorded.*"))
        for step_taken, result_of_step in final_state['past_steps']:
            # Result_of_step is already expected to be Markdown formatted
            console.print(Markdown(f"- **Step:** {step_taken}\n  **Result:**\n{result_of_step if result_of_step.strip() else '*No output from this step.*'}"))
            console.print("") # Spacer between history items
    else:
        console.print(Markdown("*No steps were recorded in the execution history.*"))
    console.rule(style="bold green")
