import os # Already imported, but ensure it's used for getcwd
import json
import argparse
import functools
import sys # Added for sys.exit
import subprocess # Added for shell tool
from langchain.tools import Tool # Added for shell tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated

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
                success_msg = f"Changed virtual working directory to: {AGENT_WORKING_DIRECTORY}"
                print(success_msg)
                return success_msg
            else:
                error_msg = f"Error: Directory not found or not a directory: {new_dir} (from CWD: {AGENT_WORKING_DIRECTORY})"
                print(error_msg)
                return error_msg

        # For other commands, execute in AGENT_WORKING_DIRECTORY
        print(f"Attempting to execute shell command: '{command}' in directory '{AGENT_WORKING_DIRECTORY}'")
        process = subprocess.run(
            command,
            shell=True, # Still using shell=True for convenience; be mindful of security.
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=AGENT_WORKING_DIRECTORY # Use the agent's current working directory
        )

        output = ""
        if process.stdout:
            output += f"Stdout:\n{process.stdout.strip()}"
        if process.stderr:
            if process.stdout.strip(): # Check if stdout had actual content
                output += "\n"
            output += f"Stderr:\n{process.stderr.strip()}"

        if not output.strip() and process.returncode == 0: # Check if output is effectively empty
            output = "Command executed successfully with no output."
        elif not output.strip() and process.returncode != 0:
            output = f"Command failed with return code {process.returncode} and no output."


        full_log = f"Command: '{command}', CWD: '{AGENT_WORKING_DIRECTORY}', Return Code: {process.returncode}, Output:\n{output.strip()}"
        print(full_log)
        return output.strip()

    except subprocess.TimeoutExpired:
        error_message = f"Error: Command '{command}' timed out after 30 seconds (CWD: {AGENT_WORKING_DIRECTORY})."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error executing command '{command}' (CWD: {AGENT_WORKING_DIRECTORY}): {e}"
        print(error_message)
        return error_message

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

    print(f"\n--- Executing Step {state['next_step_index'] + 1} ---")
    print(f"Action: {current_step_description}")

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
                    print(f"EXECUTOR: Identified shell command: '{command}' for step: '{current_step_description}' (Tool: {shell_tool.name})")
                    step_output = shell_tool.run(command) # Execute the command.
                elif command: # Command is present but not a string.
                     error_msg = f"Error: {shell_tool.name} was called, but 'command_to_run' was not a valid string: Got '{command}' (type: {type(command).__name__})."
                     print(error_msg)
                     step_output = error_msg
                else: # Command is missing.
                    error_msg = f"Error: {shell_tool.name} was called, but no 'command_to_run' was provided."
                    print(error_msg)
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
            print(f"Error processing executor response or during tool call attempt: {e}")
            step_output = f"Error during step execution logic: {e}"

    except Exception as e: # Catch errors from the LLM call itself.
        print(f"Error during executor LLM call for step '{current_step_description}': {e}")
        step_output = f"Error executing step (LLM call failed): {e}"

    print(f"Result: {step_output}")
    print("--------------------------")

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

    except Exception as e:
        print(f"Error during planner LLM call or plan parsing: {e}")
        # Return a state that indicates failure
        generated_plan = [f"Failed to generate a plan due to an error: {e}"]
        return {"plan": generated_plan, "next_step_index": 0, "past_steps": []}

    print("\n--- Generated Plan ---")
    for i, step_item in enumerate(generated_plan):
        print(f"{i+1}. {step_item}")
    print("----------------------")

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
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        sys.exit(1)

def read_directory_content(dir_path: str) -> str:
    """Recursively reads content of all files in a directory."""
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found at {dir_path}")
        sys.exit(1)

    all_content = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                file_content = read_file_content(filepath)
                all_content.append(f"\n--- Content from {filepath} ---\n{file_content}")
            except Exception as e: # Catching generic exception from read_file_content
                print(f"Skipping file {filepath} due to error: {e}")
                # Optionally, decide if this should halt execution or just skip the file
    return "\n".join(all_content)

# 8. Main Execution Logic
if __name__ == "__main__":
    # ---- ADD WARNING START ----
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: Buddy AI Agent - Shell Command Execution      !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("This agent can execute shell commands generated by an AI model.")
    print("Executing AI-generated commands can be DANGEROUS and may lead to:")
    print("- Unintended system modifications")
    print("- Data loss or corruption")
    print("- Security vulnerabilities")
    print("ALWAYS review the generated plan and the specific commands before execution if possible,")
    print("and only run this agent in a safe, isolated environment if you are unsure.")
    print("You are responsible for any actions taken by this agent.")
    print("--------------------------------------------------------------")
    # ---- ADD WARNING END ----

    # API Key Check
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY is not set. Please set it in the script or as an environment variable.")
        sys.exit(1)

    # Initialize LLM after key check
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
    except Exception as e:
        print(f"Error initializing LLM (gemini-2.5-flash-preview-04-17): {e}")
        print("Attempting to initialize with 'gemini-pro' as a fallback...")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
        except Exception as e_pro:
            print(f"Error initializing LLM (gemini-pro): {e_pro}")
            print("LLM initialization failed. Please check your API key and model availability.")
            sys.exit(1)

    parser = argparse.ArgumentParser(description="Buddy Agent: A helpful AI assistant.")
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
            print(f"Reading prompt from file: {prompt_input}")
            user_prompt = read_file_content(prompt_input)
        else:
            user_prompt = prompt_input
    except Exception as e:
        print(f"Failed to read prompt: {e}")
        exit(1)

    # Process Context
    context_input = args.context
    context_data = ""
    if context_input:
        print(f"Loading context from: {context_input}")
        try:
            if os.path.isfile(context_input):
                context_data = read_file_content(context_input)
            elif os.path.isdir(context_input):
                context_data = read_directory_content(context_input)
            else:
                print(f"Warning: Context path '{context_input}' is not a valid file or directory. Proceeding without context.")
        except Exception as e:
            print(f"Failed to load context from {context_input}: {e}. Proceeding without context.")
            context_data = "" # Ensure context_data is empty on error

    # Prepare Initial State for LangGraph
    initial_state = {
        "prompt": user_prompt,
        "context": context_data,
        # plan, past_steps, next_step_index are set by the graph
    }

    # User feedback
    print("\n===================================")
    print("    BUDDY AGENT INITIALIZING     ")
    print("===================================")
    print(f"\nUser Prompt: {user_prompt}")
    context_preview = context_data[:200] + '...' if len(context_data) > 200 else context_data # Shorter preview
    print(f"Context Provided: {'Yes' if context_data else 'No'}")
    if context_data:
        # Ensure context_preview is defined here if you want to use it.
        # For simplicity, just confirming it's loaded.
        print(f"Context (Preview): {context_data[:200] + '...' if len(context_data) > 200 else context_data}")

    print("\n--- Starting Agent Execution ---")

    # Invoke the agent. The plan and step-by-step execution will be printed by the nodes.
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        print(f"\n--- Agent Execution Failed ---")
        print(f"An unexpected error occurred during agent execution: {e}")
        # Depending on the error, parts of final_state might be available or it might be None
        # We'll try to print what we can, or a generic message
        final_state = {} # Ensure final_state is a dict for the summary below
        # Potentially add more specific error state updates here if needed

    print("\n\n===================================")
    print("    AGENT EXECUTION COMPLETE     ")
    print("===================================")

    print("\n--- Final Summary ---")
    print("Final Plan Executed:")
    if 'plan' in final_state and final_state['plan']:
        for i, step_content in enumerate(final_state['plan']):
            print(f"  {i+1}. {step_content}")
    else:
        print("  No plan was generated or available in the final state.")

    print("\nExecution History (Step: Result):")
    if 'past_steps' in final_state and final_state['past_steps']:
        if not final_state['past_steps']:
            print("  No steps were executed.")
        for step_taken, result_of_step in final_state['past_steps']:
            # Ensure result_of_step is a string and handle potential multi-line results
            result_str = str(result_of_step).replace('\n', '\n      ')
            print(f"  - Step: {step_taken}")
            print(f"    Result: {result_str}")
    else:
        print("  No execution history was recorded.")
    print("===================================")
