# buddy.py
import os
import argparse
import pathlib
from typing import TypedDict, List, Optional, Annotated
import operator
import traceback

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool
from langgraph.graph import StateGraph, END
# ChatPromptTemplate not strictly needed for this version as prompts are directly formatted strings
from pydantic import BaseModel, Field # Updated import for Pydantic v2
from langchain_core.messages import HumanMessage

from langgraph.prebuilt import create_react_agent

DEFAULT_API_KEY = "AIzaSyAAfE6ydHeGx9-VVVVMbBLcMrB8QtGdpfE"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", DEFAULT_API_KEY)
if GOOGLE_API_KEY == DEFAULT_API_KEY:
    print("Using default GOOGLE_API_KEY from script.")
else:
    print("Using GOOGLE_API_KEY from environment.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("Buddy AI Agent Initializing...")

def read_file_or_directory(path_str: str) -> str:
    path = pathlib.Path(path_str)
    content_parts = []
    if path.is_file():
        try:
            content_parts.append(f"--- Content from file: {path.name} ---\n{path.read_text(encoding='utf-8', errors='ignore')}")
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return ""
    elif path.is_dir():
        content_parts.append(f"--- Content from directory: {path.name} ---")
        found_files = False
        allowed_extensions = [
            ".txt", ".py", ".md", ".sh", ".json", ".yaml", ".yml", ".h", ".c", ".cc", ".cpp", ".java",
            ".js", ".ts", ".html", ".css", ".rb", ".php", ".pl", ".tcl", ".go", ".rs", ".swift",
            ".kt", ".scala", ".r", ".ps1", ".psm1", ".bat", ".cmd", ".vb", ".vbs", ".sql", ".xml",
            ".ini", ".cfg", ".conf", ".toml", ".dockerfile", "Dockerfile", ".tf"
        ]
        for item in path.rglob("*"):
            if item.is_file() and (item.suffix.lower() in allowed_extensions or item.name == "Dockerfile"):
                try:
                    content_parts.append(f"--- Content from file: {item.relative_to(path)} ---\n{item.read_text(encoding='utf-8', errors='ignore')}")
                    found_files = True
                except Exception as e:
                    print(f"Error reading file {item}: {e}")
        if not found_files:
             content_parts.append("No text files found in directory.")
    else:
        print(f"Error: Path '{path_str}' is not a valid file or directory.")
        return ""
    return "\n\n".join(content_parts)

def create_llm_instance(model_name_primary: str, model_name_fallback: str, llm_type: str):
    try:
        llm = ChatGoogleGenerativeAI(model=model_name_primary, temperature=0, convert_system_message_to_human=True)
        print(f"{llm_type} LLM created successfully using {model_name_primary}.")
        return llm
    except Exception as e:
        print(f"Error creating {llm_type} LLM with {model_name_primary}: {e}. Trying fallback {model_name_fallback}.")
        try:
            llm = ChatGoogleGenerativeAI(model=model_name_fallback, temperature=0, convert_system_message_to_human=True)
            print(f"{llm_type} LLM created successfully using {model_name_fallback} (fallback).")
            return llm
        except Exception as e_fallback:
            print(f"Error creating {llm_type} LLM with fallback {model_name_fallback}: {e_fallback}")
            traceback.print_exc()
            return None

def create_executor_agent_runnable(llm):
    if not llm: return None
    tools = [ShellTool()]
    try:
        executor_agent = create_react_agent(llm, tools=tools)
        print("Executor ReAct agent created successfully.")
        return executor_agent
    except Exception as e:
        print(f"Error creating Executor Agent: {e}")
        traceback.print_exc()
        return None

class Plan(BaseModel):
    # No docstring here to avoid parsing issues
    steps: List[str] = Field(description="Actionable steps for the executor.") # Simplified description

class BuddyGraphState(TypedDict):
    objective: str
    context: Optional[str]
    plan: Optional[List[str]]
    current_step_index: int
    step_results: Annotated[List[str], operator.add]
    final_output: Optional[str]

_planner_llm_structured_global = None
_executor_agent_global = None

_PLANNER_PROMPT_TEMPLATE_STR = (
    "You are a master planning agent. Create a detailed, step-by-step plan to achieve the user objective. "
    "Use the provided context. Each step MUST be a clear, atomic, actionable instruction for an executor with a shell tool. "
    "For coding: write code, save code, run code, check output. For analysis: list files, read files, etc. "
    "User Objective: {objective} Context (if any):\n{context} " # Ensure context is handled if empty
    "Respond ONLY with the structured plan."
)

def planner_node_func(state: BuddyGraphState):
    print("\n--- Planner Node ---")
    objective = state["objective"]
    context_str = state.get("context") if state.get("context") is not None else ""

    global _planner_llm_structured_global
    if not _planner_llm_structured_global:
        print("CRITICAL ERROR: Planner LLM not initialized.")
        return {"plan": ["Critical Error: Planner LLM not initialized."], "current_step_index": 0, "step_results": []}

    formatted_prompt = _PLANNER_PROMPT_TEMPLATE_STR.format(objective=objective, context=context_str)
    print(f"Planner input (first 300 chars): {formatted_prompt[:300]}...")

    plan_steps_list = []
    try:
        ai_response = _planner_llm_structured_global.invoke(formatted_prompt)
        if ai_response and hasattr(ai_response, 'steps') and isinstance(ai_response.steps, list) and ai_response.steps:
            plan_steps_list = ai_response.steps
        else:
            plan_steps_list = ["Planner returned no steps or invalid structure."]
            print(f"Invalid plan structure from LLM: {ai_response}")
        print(f"Generated plan: {plan_steps_list}")
    except Exception as e:
        print(f"Error invoking structured planner LLM: {e}")
        traceback.print_exc()
        plan_steps_list = [f"Error in planning: {str(e)}"]

    return {"plan": plan_steps_list, "current_step_index": 0, "step_results": []}

def executor_node_func(state: BuddyGraphState):
    print("\n--- Executor Node ---")
    global _executor_agent_global
    if not _executor_agent_global:
        print("CRITICAL ERROR: Executor agent not initialized.")
        return {"step_results": ["Critical Error: Executor agent not initialized."], "current_step_index": state.get("current_step_index", 0) + 1}

    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)
    step_output_str = "Executor error: Pre-execution."
    next_idx = current_idx + 1

    if not plan or not isinstance(plan, list) or current_idx >= len(plan) or not plan[current_idx]:
        step_output_str = "Executor error: Invalid plan or step."
    else:
        current_instruction = plan[current_idx]
        print(f"Executing step {current_idx + 1}/{len(plan)}: {current_instruction}")
        try:
            agent_input_dict = {"messages": [HumanMessage(content=current_instruction)]}
            agent_response_dict = _executor_agent_global.invoke(agent_input_dict)
            if agent_response_dict and "messages" in agent_response_dict and agent_response_dict["messages"]:
                step_output_str = str(agent_response_dict["messages"][-1].content)
            else:
                step_output_str = "No response or unexpected format from executor."
        except Exception as e:
            print(f"Error invoking executor for step '{current_instruction}': {e}")
            traceback.print_exc()
            step_output_str = f"Error executing step: {str(e)}"
    print(f"Step {current_idx + 1} output: {step_output_str}")
    return {"step_results": [step_output_str], "current_step_index": next_idx}

def route_decider_func(state: BuddyGraphState):
    current_idx = state.get('current_step_index', 0)
    plan_steps = state.get('plan')
    if not plan_steps or not isinstance(plan_steps, list) or current_idx >= len(plan_steps) or (plan_steps and plan_steps[0].startswith("Critical Error")): # Added check for planner error
        return "end_workflow"
    return "continue_to_executor"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buddy AI Agent")
    parser.add_argument("--prompt", type=str, required=True, help="User instructions (string or filepath)")
    parser.add_argument("--context", type=str, help="Context information (filepath or directory)")
    args = parser.parse_args()

    main_prompt_content = args.prompt
    if pathlib.Path(args.prompt).is_file():
        try:
            main_prompt_content = pathlib.Path(args.prompt).read_text(encoding='utf-8')
        except Exception as e: exit(f"Error reading prompt file: {e}")

    main_context_content = ""
    if args.context: main_context_content = read_file_or_directory(args.context)

    print(f"--- Prompt ---\n{main_prompt_content}")
    print(f"--- Context ---\n{main_context_content if main_context_content else 'None'}")

    planner_llm = create_llm_instance("gemini-1.5-flash-latest", "gemini-pro", "Planner")
    if not planner_llm: exit("CRITICAL: Planner LLM creation failed.")
    try:
        _planner_llm_structured_global = planner_llm.with_structured_output(Plan)
    except Exception as e: exit(f"CRITICAL: Failed to attach structured output to planner: {e}")

    # Changed primary model for executor to a known working one
    executor_llm = create_llm_instance("gemini-1.5-flash-latest", "gemini-pro", "Executor")
    if not executor_llm: exit("CRITICAL: Executor LLM creation failed.")
    _executor_agent_global = create_executor_agent_runnable(executor_llm)
    if not _executor_agent_global: exit("CRITICAL: Executor agent creation failed.")

    workflow_graph = StateGraph(BuddyGraphState)
    workflow_graph.add_node("planner", planner_node_func)
    workflow_graph.add_node("executor", executor_node_func)
    workflow_graph.set_entry_point("planner")
    workflow_graph.add_conditional_edges("planner", route_decider_func, {"continue_to_executor": "executor", "end_workflow": END})
    workflow_graph.add_conditional_edges("executor", route_decider_func, {"continue_to_executor": "executor", "end_workflow": END})

    app_runnable = workflow_graph.compile()
    print("Graph compiled.")

    initial_graph_state = {
        "objective": main_prompt_content, "context": main_context_content,
        "current_step_index": 0, "step_results": [], "plan": None, "final_output": None
    }

    print("\n--- Invoking Graph ---")
    try:
        for i, event in enumerate(app_runnable.stream(initial_graph_state, {"recursion_limit": 25})):
            print(f"--- Stream Event {i+1} ---")
            for node_name, state_data in event.items():
                print(f" Node '{node_name}':")
                if isinstance(state_data, dict):
                    for key, val in state_data.items():
                        val_repr = str(val)[:150] + ('...' if len(str(val)) > 150 else '')
                        print(f"    {key}: {val_repr}")
        final_graph_state = app_runnable.invoke(initial_graph_state, {"recursion_limit": 25})
    except Exception as e:
        print(f"Error during graph invocation: {e}")
        traceback.print_exc()
        final_graph_state = None

    if final_graph_state:
        print("\n--- Final Graph State ---")
        if final_graph_state.get('plan'): print(f"Plan: {final_graph_state['plan']}")
        if final_graph_state.get('step_results'):
            print("Results:")
            for res in final_graph_state['step_results']: print(f"  - {res}")
            print("\n--- Consolidated Output ---")
            print("\n---\n".join(map(str,final_graph_state['step_results'])))
    else:
        print("\nGraph execution failed or produced no final state.")
    print("\nBuddy application finished.")
