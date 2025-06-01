import logging
import operator
from typing import TypedDict, List, Optional, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
# Rich console is managed by CLI module

# --- Pydantic Models and TypedDicts for Graph State ---
class Plan(BaseModel):
    """A plan consisting of a list of actionable steps."""
    steps: List[str] = Field(description="Actionable steps for the executor to follow.")

class BuddyGraphState(TypedDict):
    """Represents the state of the Buddy AI graph."""
    objective: str
    context: Optional[str]
    plan: Optional[List[str]]
    current_step_index: int
    step_results: Annotated[List[str], operator.add]
    final_output: Optional[str]

# --- Global Variables for LLMs and Agent (to be initialized by CLI) ---
_planner_llm_structured: Optional[ChatGoogleGenerativeAI] = None
_executor_agent_runnable_global: Optional[callable] = None

# --- Planner Prompt Template ---
_PLANNER_PROMPT_TEMPLATE = (
    "You are a master planning agent. Your sole task is to create a detailed, step-by-step plan "
    "to achieve the user's stated objective. You have access ONLY to a `ShellTool` that can execute general shell commands. "
    "Therefore, every step in your plan that involves interaction with the system (creating files, running scripts, listing content, etc.) "
    "MUST be a precise, complete, and directly executable shell command string, or a clear instruction that the executor can turn into one. "
    "The execution agent will pass this command string directly to the shell. "
    "User Objective: {objective}\n"
    "Context (if any):\n{context}\n\n"
    "Key Guidelines for Plan Steps:\n"
    "1.  **Shell Command Syntax:** Each action step must be a valid shell command. E.g., `ls -la`.\n"
    "2.  **Reporting Outputs:** If a step's purpose is to retrieve information, the plan must include an explicit instruction for the executor to report that information. E.g., `Execute 'cat file.txt' and report its content.`\n"
    "3.  **File Creation/Writing:** Use `echo` or `printf` with proper quoting. E.g., `printf 'First line.\\nSecond line.' > /path/to/file.txt`. Ensure directories exist: `mkdir -p /path/to/dir && echo 'content' > /path/to/dir/file.txt`.\n"
    "4.  **Script Execution:** E.g., `python /script.py`. Report script's output.\n"
    "5.  **Verification:** Consider adding steps to verify changes, e.g., `cat /path/to/file.txt`.\n"
    "6.  **Atomicity:** Each step should ideally be a single, atomic shell command.\n"
    "7.  **Quoting:** Pay close attention to shell quoting rules for paths and content with spaces/special characters.\n\n"
    "Respond ONLY with the structured plan. The execution agent is responsible for interpreting these steps."
    "Example: `Create /tmp/my_app then write 'Hello World' into /tmp/my_app/greeting.txt using mkdir and echo.`"
    "Direct style: `Execute shell command: ls -l /tmp`"
)

# --- LLM and Agent Creation Functions ---
def create_llm_instance(model_name: str, llm_type: str, api_key: Optional[str] = None) -> Optional[ChatGoogleGenerativeAI]:
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        logging.info(f"{llm_type} LLM created successfully using {model_name}.")
        return llm
    except Exception as e:
        logging.error(f"Error creating {llm_type} LLM with {model_name}: {e}", exc_info=True)
        return None

def create_executor_agent_runnable(llm: ChatGoogleGenerativeAI) -> Optional[callable]:
    try:
        executor_agent = create_react_agent(llm, tools=[ShellTool()])
        logging.info("Executor ReAct agent created successfully with ShellTool.")
        return executor_agent
    except Exception as e:
        logging.error(f"Error creating Executor Agent: {e}", exc_info=True)
        return None

# --- Graph Nodes ---
def planner_node(state: BuddyGraphState) -> dict:
    logging.info("Entering planner_node.")
    objective = state["objective"]
    context = state.get("context", "")

    global _planner_llm_structured
    if not _planner_llm_structured:
        logging.error("Planner LLM (_planner_llm_structured) is not initialized in planner_node.")
        return {"plan": ["Critical Error: Planner LLM not initialized."], "current_step_index": 0, "step_results": []}

    formatted_prompt = _PLANNER_PROMPT_TEMPLATE.format(objective=objective, context=context if context else "No context provided.")
    logging.debug(f"Planner input prompt: {formatted_prompt}")

    plan_steps = ["Critical Error: Planner failed to generate a plan."]
    try:
        ai_response = _planner_llm_structured.invoke(formatted_prompt)
        if ai_response and isinstance(ai_response, Plan) and ai_response.steps:
            plan_steps = ai_response.steps
            if not all(isinstance(step, str) for step in plan_steps):
                logging.error(f"Invalid plan structure (non-string steps): {ai_response.steps}")
                plan_steps = ["Critical Error: Planner returned non-string steps."]
        else:
            logging.error(f"Invalid plan structure or empty plan from LLM: {ai_response}")
            plan_steps = ["Critical Error: Planner returned no steps or invalid plan structure."]
        logging.info(f"Generated plan: {plan_steps}")
    except Exception as e:
        logging.error(f"Error invoking structured planner LLM: {e}", exc_info=True)
        plan_steps = [f"Critical Error: Exception during planning - {str(e)}"]
    return {"plan": plan_steps, "current_step_index": 0, "step_results": []}

def executor_node(state: BuddyGraphState) -> dict:
    current_idx = state.get("current_step_index", 0)
    logging.info(f"Entering executor_node for step index {current_idx}.")

    global _executor_agent_runnable_global
    if not _executor_agent_runnable_global:
        logging.error("Executor agent (_executor_agent_runnable_global) is not initialized.")
        return {"step_results": ["Critical Error: Executor agent not initialized."], "current_step_index": current_idx + 1}

    plan = state.get("plan")
    step_output_str = "Error: Pre-execution state error in executor_node."
    next_step_idx = current_idx + 1

    if not plan or not isinstance(plan, list) or not (0 <= current_idx < len(plan)) or not plan[current_idx]:
        step_output_str = "Error: Invalid plan, step index out of bounds, or empty step content."
        logging.error(step_output_str)
        if plan and isinstance(plan, list) and len(plan) > 0 and plan[0].startswith("Critical Error"):
            step_output_str = plan[0]
    else:
        current_instruction = plan[current_idx]
        logging.info(f"Executing step {current_idx + 1}/{len(plan)}: {current_instruction}")
        try:
            agent_input = {"messages": [HumanMessage(content=current_instruction)]}
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
    logging.info(f"Step {current_idx + 1} output: {step_output_str}")
    return {"step_results": [step_output_str], "current_step_index": next_step_idx}

def should_continue_decider(state: BuddyGraphState) -> str:
    logging.info("Entering should_continue_decider.")
    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)

    if not plan or not isinstance(plan, list) or current_idx >= len(plan):
        if current_idx >= len(plan) and plan and len(plan) > 0:
            logging.info("Decider: All steps executed or current index exceeds plan length. Ending workflow.")
        else:
            logging.info("Decider: Invalid or empty plan. Ending workflow.")
        return "end_workflow"
    if plan[0].startswith("Critical Error:"):
        logging.info("Decider: Critical error in plan from planner node. Ending workflow.")
        return "end_workflow"
    step_results = state.get("step_results", [])
    if step_results and step_results[-1].startswith("Critical Error:"):
        logging.info("Decider: Critical error from executor node. Ending workflow.")
        return "end_workflow"
    logging.info(f"Decider: Continuing to execute. Next step index: {current_idx}.")
    return "continue_to_executor"

# --- Graph Setup ---
def set_global_llms_and_agents(planner_llm: ChatGoogleGenerativeAI, executor_agent: callable):
    global _planner_llm_structured, _executor_agent_runnable_global
    try:
        _planner_llm_structured = planner_llm.with_structured_output(Plan)
        logging.info("Global planner LLM configured for structured output (Plan).")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to configure global planner LLM for structured output: {e}", exc_info=True)
        raise
    _executor_agent_runnable_global = executor_agent
    logging.info("Global executor agent runnable set.")

def create_buddy_graph() -> StateGraph:
    logging.info("Defining StateGraph workflow...")
    workflow = StateGraph(BuddyGraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner", should_continue_decider,
        {"continue_to_executor": "executor", "end_workflow": END}
    )
    workflow.add_conditional_edges(
        "executor", should_continue_decider,
        {"continue_to_executor": "executor", "end_workflow": END}
    )
    app = workflow.compile()
    logging.info("StateGraph compiled successfully.")
    return app
