import json
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from rich.console import Console # Will be used for type hint or if an instance is passed
from rich.markdown import Markdown
from buddy_agent.utils import parse_llm_list_output # Import from utils

# Define Agent State
class PlanExecuteState(TypedDict):
    prompt: str
    plan: List[str]
    past_steps: List[tuple[str, str]]
    context: str
    next_step_index: int

# LLM should be passed to functions requiring it, or initialized globally if appropriate.
# For now, get_planner and get_executor will take llm as an argument.
# The global `llm` from buddy.py will be handled in cli/main.py

# --- Functions related to graph definition and execution ---

# parse_llm_list_output has been moved to buddy_agent.utils

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
    return planner_chain

# Executor needs access to shell_tool, which will be in tools/shell.py
# This creates a dependency. For now, we'll assume shell_tool is imported or passed.
from buddy_agent.tools.shell import shell_tool # Ensure this is correctly placed

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
    return executor_chain

# Global LLM needs to be accessible to plan_step and execute_step.
# This will be initialized in cli.main and passed or set.
# For now, these functions will require `llm` to be in their scope.
# A better way would be to pass llm as an argument to these graph steps if they are methods of a class,
# or configure the graph nodes with the llm instance.
# For now, let's assume `llm` will be available in the scope where these are called by the graph.
_llm_instance = None # Placeholder for LLM instance
_console_instance = None # Placeholder for Console instance

def configure_graph_dependencies(llm, console_obj: Console):
    global _llm_instance, _console_instance
    _llm_instance = llm
    _console_instance = console_obj

def plan_step(state: PlanExecuteState):
    if not _llm_instance or not _console_instance:
        raise ValueError("LLM or Console instance not set in graph.py. Call configure_graph_dependencies() first.")
    planner_chain = get_planner(_llm_instance)
    generated_plan: List[str] = []
    try:
        response = planner_chain.invoke({"prompt": state["prompt"], "context": state["context"]})
        plan_str = response.content if hasattr(response, 'content') else str(response)
        generated_plan = parse_llm_list_output(plan_str) # Imported from utils
        if not generated_plan:
            _console_instance.print(Markdown("*Planner Warning: No actionable steps identified or plan was empty.*"))
            generated_plan = ["No actionable steps identified by the planner or plan was empty."]
    except Exception as e:
        error_msg = f"Error during planner LLM call or plan parsing: {e}"
        _console_instance.print(Markdown(f"# Plan Generation Failed\n\n**Error:** {error_msg}"))
        generated_plan = [f"Failed to generate a plan due to an error: {e}"]

    _console_instance.rule("[bold cyan]Execution Plan")
    if generated_plan and not (len(generated_plan) == 1 and "Failed to generate a plan" in generated_plan[0]):
        markdown_plan_str = "\n".join([f"{i+1}. {step}" for i, step in enumerate(generated_plan)])
        _console_instance.print(Markdown(markdown_plan_str))
    elif not generated_plan:
        _console_instance.print(Markdown("*No plan steps were generated.*"))
    _console_instance.print("")

    return {"plan": generated_plan, "next_step_index": 0, "past_steps": []}

def execute_step(state: PlanExecuteState):
    if not _llm_instance or not _console_instance:
        raise ValueError("LLM or Console instance not set in graph.py. Call configure_graph_dependencies() first.")
    executor_chain = get_executor(_llm_instance)
    current_step_description = state['plan'][state['next_step_index']]
    plan_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state['plan']))
    past_steps_formatted = "\n".join([f"Step: {ps[0]}\nResult: {ps[1]}" for ps in state['past_steps']]) \
        if state['past_steps'] else "No steps executed yet."
    step_number = state['next_step_index'] + 1
    rule_title = f"[bold cyan]Executing Step {step_number}: {current_step_description[:70]}{'...' if len(current_step_description) > 70 else ''}"
    _console_instance.rule(rule_title)

    step_output: str
    try:
        raw_executor_response_message = executor_chain.invoke({
            "prompt": state['prompt'],
            "context": state['context'],
            "plan_str": plan_str,
            "past_steps_str": past_steps_formatted,
            "current_step": current_step_description
        })
        raw_executor_response = raw_executor_response_message.content if hasattr(raw_executor_response_message, 'content') else str(raw_executor_response_message)

        try:
            cleaned_response = raw_executor_response.strip()
            if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("`json") and cleaned_response.endswith("`"):
                 cleaned_response = cleaned_response[5:-1].strip()

            tool_call_request = json.loads(cleaned_response)
            if isinstance(tool_call_request, dict) and tool_call_request.get("tool_to_use") == shell_tool.name:
                command = tool_call_request.get("command_to_run")
                if command and isinstance(command, str):
                    _console_instance.print(Markdown(f"Executing shell command: `{command}`"))
                    # shell_tool.run itself uses a local console for its internal logging.
                    step_output = shell_tool.run(command)
                elif command:
                     error_msg = f"Error: {shell_tool.name} was called, but 'command_to_run' was not a valid string: Got '{command}' (type: {type(command).__name__})."
                     _console_instance.print(Markdown(f"**Shell Tool Error:** {error_msg}"))
                     step_output = error_msg
                else:
                    error_msg = f"Error: {shell_tool.name} was called, but no 'command_to_run' was provided."
                    _console_instance.print(Markdown(f"**Shell Tool Error:** {error_msg}"))
                    step_output = error_msg
            else:
                step_output = raw_executor_response
        except json.JSONDecodeError:
            # The response was not valid JSON. Treat as a direct answer from the LLM.
            step_output = raw_executor_response
        except Exception as e:
            # Catch any other errors during the parsing or tool logic.
            error_text = f"Error processing executor response or during tool call attempt: {e}"
            _console_instance.print(Markdown(f"**Execution Error:** {error_text}"))
            step_output = f"Error during step execution logic: {e}"

    except Exception as e: # Catch errors from the LLM call itself.
        error_text = f"Error during executor LLM call for step '{current_step_description}': {e}"
        _console_instance.print(Markdown(f"**LLM Call Error:** {error_text}"))
        step_output = f"Error executing step (LLM call failed): {e}"

    _console_instance.print(Markdown("**Result:**"))
    _console_instance.print(Markdown(step_output if step_output.strip() else "*No output from step.*"))
    _console_instance.print("")

    updated_past_steps = state['past_steps'] + [(current_step_description, step_output)]
    next_index = state['next_step_index'] + 1
    return {"past_steps": updated_past_steps, "next_step_index": next_index}

def should_continue(state: PlanExecuteState):
    if state['next_step_index'] < len(state['plan']):
        return "execute"
    return "end"

# Create the Graph
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

# Function to get the compiled app
def get_graph_app():
    return app

    return {"past_steps": updated_past_steps, "next_step_index": next_index}

def should_continue(state: PlanExecuteState):
    if state['next_step_index'] < len(state['plan']):
        return "execute"
    return "end"

# Create the Graph
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

# Function to get the compiled app
def get_graph_app():
    return app

# configure_graph_dependencies replaces configure_graph_llm and sets console
# Example of how this might be used (for clarity, not for execution here):
# if __name__ == '__main__':
#     # This part would typically be in your main CLI script
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     # Dummy LLM and Console for demonstration
#     my_llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="YOUR_API_KEY")
#     my_console = Console()
#     configure_graph_dependencies(my_llm, my_console)
#
#     # Now the graph 'app' can be invoked with an initial state
#     initial_state_example = {
#         "prompt": "List files in current directory and then write hello to a file named world.txt",
#         "context": "",
#     }
#     result = app.invoke(initial_state_example)
#     _console_instance.print(result)
