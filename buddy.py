import os
import json
import argparse
import functools
import sys # Added for sys.exit
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

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful AI assistant that generates a step-by-step plan to address a user's request. "
         "The user's request is provided in the 'prompt' and any relevant 'context' is also given. "
         "Break down the prompt into a series of clear, numbered steps. "
         "Output *only* the list of steps, with each step on a new line. Do not include any other text or explanations. "
         "For example, if the prompt is 'Research and write a blog post about AI', your output should be like:\n"
         "1. Research the current trends in AI.\n"
         "2. Outline the structure of the blog post.\n"
         "3. Write the first draft of the blog post.\n"
         "4. Review and edit the blog post.\n"
         "5. Publish the blog post."),
        ("human", "Prompt: {prompt}\nContext: {context}")
    ]
)

def get_planner(llm):
    planner_chain = PLANNER_PROMPT | llm
    # The content extraction and parsing will be handled in plan_step
    return planner_chain

# 5. Implement Executor Node
EXECUTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful AI assistant that executes a single step from a plan. "
         "You are given the overall 'prompt', the 'context', the 'plan' (list of all steps), "
         "the 'past_steps' (already executed steps and their results), and the 'current_step' to execute. "
         "Focus *only* on the 'current_step'. Do not try to execute other steps. "
         "Provide a concise result for the current step. Do not output any other text or explanations. "
         "If the current step is to write code, provide only the code. "
         "If the current step is to answer a question, provide only the answer."),
        ("human",
         "Overall Prompt: {prompt}\n"
         "Context: {context}\n"
         "Full Plan:\n{plan_str}\n"
         "Past Steps (step, result):\n{past_steps_str}\n"
         "Current Step to Execute: {current_step}")
    ]
)

def get_executor(llm):
    executor_chain = EXECUTOR_PROMPT | llm
    return executor_chain | (lambda ऐresponse: ऐresponse.content) # Extract content from AIMessage

# 6. Define Graph Logic Functions
def execute_step(state: PlanExecuteState):
    executor = get_executor(llm) # llm should be initialized by now
    current_step_description = state['plan'][state['next_step_index']]

    plan_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(state['plan']))
    past_steps_str = "\n".join(f"- {step}: {result}" for step, result in state['past_steps']) if state['past_steps'] else "No steps executed yet."

    print(f"\n--- Executing Step {state['next_step_index'] + 1} ---")
    print(f"Action: {current_step_description}")

    llm_result: str
    try:
        llm_result = executor.invoke({
            "prompt": state['prompt'],
            "context": state['context'],
            "plan_str": plan_str,
            "past_steps_str": past_steps_str,
            "current_step": current_step_description
        })
    except Exception as e:
        print(f"Error during executor LLM call for step '{current_step_description}': {e}")
        llm_result = f"Error executing step: {e}" # Store error as result

    print(f"Result: {llm_result}")
    print("--------------------------")

    updated_past_steps = state['past_steps'] + [(current_step_description, llm_result)]
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
