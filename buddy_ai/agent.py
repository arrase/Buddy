import logging
import operator
from typing import TypedDict, List, Optional, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from rich.markdown import Markdown # Added for printing plan
from rich.console import Console # Added for agent console


# --- Pydantic Models and TypedDicts for Graph State ---
class Plan(BaseModel):
    """A plan consisting of a list of actionable steps."""
    steps: List[str] = Field(description="Actionable steps for the executor to follow.")

class Assessment(BaseModel): # NEW
    objective_met: bool = Field(description="Whether the user's objective has been fully met.")
    reasoning: str = Field(description="Brief explanation for why the objective is considered met or not.")
    suggest_replanning: bool = Field(description="If objective is not met, should we try to replan? Set to false if errors are unrecoverable or further attempts seem futile.")

class BuddyGraphState(TypedDict):
    """Represents the state of the Buddy AI graph."""
    objective: str
    context: Optional[str]
    plan: Optional[List[str]]
    current_step_index: int
    step_results: Annotated[List[str], operator.add]
    final_output: Optional[str]
    user_feedback: Optional[str] = None
    plan_approved: bool = False
    auto_approve: bool = False

# --- Global Variables for LLMs and Agent (to be initialized by CLI) ---
_planner_llm_structured: Optional[ChatGoogleGenerativeAI] = None
_executor_agent_runnable_global: Optional[callable] = None
_assessor_llm_global: Optional[ChatGoogleGenerativeAI] = None # NEW
_agent_cli_console: Optional[Console] = None # Added for agent console

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

# --- Assessment Prompt Template --- (NEW)
_ASSESSMENT_PROMPT_TEMPLATE = (
    "You are an assessment agent. Your task is to determine if the user's objective has been met based on the history of executed steps and their results. "
    "User Objective: {objective}\n"
    "Original Plan:\n{original_plan}\n"
    "Execution History (step-by-step results):\n{step_results_formatted}\n\n"
    "Analyze the execution history in relation to the objective. "
    "Consider if the last step's output clearly indicates success or failure, or if the overall goal is achieved. "
    "If the last step was a verification step, pay close attention to its output. "
    "If there were errors, assess if they are recoverable or if the task seems impossible with the current approach. "
    "Respond ONLY with the structured Assessment."
)

# --- Replanner Prompt Template --- (NEW)
_REPLANNER_PROMPT_TEMPLATE = (
    "You are a master replanning agent. The previous plan failed to achieve the user's objective or was insufficient. "
    "Your task is to create a NEW, detailed, step-by-step plan to achieve the original user's objective, "
    "taking into account the previous plan and the results of its execution. "
    "You have access ONLY to a `ShellTool`. Ensure all steps are precise, executable shell commands or clear instructions. "
    "Do NOT repeat steps from the previous plan that were successfully executed if they do not need to be re-done. Focus on new steps or corrections. "
    "User Objective: {objective}\n"
    "Previous Plan:\n{previous_plan}\n"
    "Execution History of Previous Plan (step-by-step results):\n{step_results_formatted}\n"
    "User Feedback for Refinement:\n{user_feedback}\n"
    "Context (if any):\n{context}\n\n"
    "Key Guidelines for Plan Steps (same as planner - ensure shell commands, reporting, verification etc.):\n"
    "1.  **Shell Command Syntax:** Each action step must be a valid shell command. E.g., `ls -la`.\n"
    "2.  **Reporting Outputs:** If a step's purpose is to retrieve information, the plan must include an explicit instruction for the executor to report that information. E.g., `Execute 'cat file.txt' and report its content.`\n"
    "3.  **File Creation/Writing:** Use `echo` or `printf` with proper quoting. E.g., `printf 'First line.\\nSecond line.' > /path/to/file.txt`. Ensure directories exist: `mkdir -p /path/to/dir && echo 'content' > /path/to/dir/file.txt`.\n"
    "4.  **Script Execution:** E.g., `python /script.py`. Report script's output.\n"
    "5.  **Verification:** Consider adding steps to verify changes, e.g., `cat /path/to/file.txt`.\n"
    "6.  **Atomicity:** Each step should ideally be a single, atomic shell command.\n"
    "7.  **Quoting:** Pay close attention to shell quoting rules for paths and content with spaces/special characters.\n\n"
    "Respond ONLY with the structured new plan."
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
    return {"plan": plan_steps, "current_step_index": 0, "step_results": [], "plan_approved": False, "user_feedback": None}

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


def replanner_node(state: BuddyGraphState) -> dict:
    logging.info("Entering replanner_node.")
    objective = state["objective"]
    previous_plan = state.get("plan", []) # Current plan is the 'previous' one
    step_results = state.get("step_results", [])
    context = state.get("context", "")

    global _planner_llm_structured # Reusing the planner's structured LLM
    if not _planner_llm_structured:
        logging.error("Replanner Node: Planner LLM (_planner_llm_structured) is not initialized.")
        return {"plan": ["Critical Error: Planner LLM not initialized for replanning."], "current_step_index": 0, "step_results": []}

    previous_plan_str = "\n".join(f"- {s}" for s in previous_plan)
    step_results_formatted = "\n".join(f"Step {i+1} Result: {res}" for i, res in enumerate(step_results))

    formatted_prompt = _REPLANNER_PROMPT_TEMPLATE.format(
        objective=objective,
        previous_plan=previous_plan_str,
        step_results_formatted=step_results_formatted,
        user_feedback=state.get("user_feedback", "No specific feedback provided."),
        context=context if context else "No additional context provided."
    )
    logging.debug(f"Replanner input prompt: {formatted_prompt}")

    new_plan_steps = ["Critical Error: Replanner failed to generate a new plan."]
    try:
        ai_response = _planner_llm_structured.invoke(formatted_prompt)
        if ai_response and isinstance(ai_response, Plan) and ai_response.steps:
            new_plan_steps = ai_response.steps
            if not all(isinstance(step, str) for step in new_plan_steps):
                logging.error(f"Invalid new plan structure (non-string steps): {ai_response.steps}")
                new_plan_steps = ["Critical Error: Replanner returned non-string steps."]
        else:
            logging.error(f"Invalid new plan structure or empty plan from LLM during replan: {ai_response}")
            new_plan_steps = ["Critical Error: Replanner returned no new steps or invalid plan structure."]
        logging.info(f"Replanner generated new plan: {new_plan_steps}")
    except Exception as e:
        logging.error(f"Error invoking structured planner LLM for replan: {e}", exc_info=True)
        new_plan_steps = [f"Critical Error: Exception during replanning - {str(e)}"]

    return {"plan": new_plan_steps, "current_step_index": 0, "user_feedback": None, "plan_approved": False}

# --- Human Approval Node ---
def human_approval_node(state: BuddyGraphState) -> dict:
    logging.info("Entering human_approval_node.")

    global _agent_cli_console
    if _agent_cli_console is None:
        logging.critical("CRITICAL: Agent CLI console not set in human_approval_node. This should have been set by the CLI.")
        # Return an error state that can be handled by the graph
        return {"plan_approved": False, "user_feedback": "Critical Error: Agent console not configured.", "plan": ["Critical Error: Agent console not configured."]}

    auto_approve = state.get("auto_approve", False)
    # Log initial state relevant to approval
    logging.info(f"Plan approval state: auto_approve={auto_approve}, current_plan_approved={state.get('plan_approved')}, user_feedback_present={bool(state.get('user_feedback'))}")

    # Initialize return state keys to avoid partial updates if logic exits early
    current_plan_approved = False
    current_user_feedback = None

    if auto_approve:
        logging.info("Auto-approving plan.")
        current_plan_approved = True
        return {"plan_approved": current_plan_approved, "user_feedback": current_user_feedback}

    # If not auto_approve, proceed with interactive approval.
    # This node now handles the input directly.
    plan = state.get("plan")

    # Check for critical errors in the plan itself before asking for approval
    if not plan or not isinstance(plan, list) or not plan[0] or plan[0].startswith("Critical Error:"):
        logging.warning(f"Critical error in plan detected by human_approval_node. Bypassing user interaction. Plan: {plan}")
        # This state will be caught by decide_after_approval to go to END
        return {"plan_approved": False, "user_feedback": None, "plan": plan} # Ensure plan is passed through


    plan_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    _agent_cli_console.print(Markdown(f"## Proposed Execution Plan:\n{plan_md}"))

    while True:
        try:
            raw_input = _agent_cli_console.input("[bold yellow]Plan Review[/bold yellow]: ([bold green]A[/bold green])pprove, ([bold blue]R[/bold blue])efine, or ([bold red]C[/bold red])ancel plan? ").strip().lower()
        except KeyboardInterrupt: # Handle Ctrl+C as cancellation
            logging.warning("User cancelled via KeyboardInterrupt during plan approval.")
            raw_input = 'c' # Treat as cancel

        if raw_input == 'a':
            logging.info("Plan approved by user.")
            current_plan_approved = True
            current_user_feedback = None
            break
        elif raw_input == 'r':
            logging.info("User chose to refine the plan.")
            while True:
                try:
                    feedback = _agent_cli_console.input("Please provide feedback for replanning: ").strip()
                    if feedback:
                        current_user_feedback = feedback
                        current_plan_approved = False
                        # logging.info(f"User feedback for replan: {feedback}") # Logged after loop breaks
                        break
                    else:
                        _agent_cli_console.print("[bold red]Feedback cannot be empty if you choose to refine. Please provide your comments or (C)ancel refinement.[/bold red]")
                        sub_choice = _agent_cli_console.input("Enter feedback or (C)ancel refinement: ").strip().lower()
                        if sub_choice == 'c':
                            current_user_feedback = None
                            break
                except KeyboardInterrupt:
                    logging.warning("User cancelled refinement input via KeyboardInterrupt.")
                    current_user_feedback = None
                    break
            if current_user_feedback is not None:
                logging.info(f"User chose to refine. Feedback: {current_user_feedback}")
                break
            # If feedback was cancelled (current_user_feedback is None), the outer loop continues

        elif raw_input == 'c':
            current_plan_approved = False
            current_user_feedback = None
            logging.info("User cancelled plan approval.")
            break
        else:
            logging.debug("Invalid input from user during plan approval.") # Log before print
            _agent_cli_console.print("[bold red]Invalid input. Please enter 'A', 'R', or 'C'.[/bold red]")

    # Log final outcome of the node before returning
    if current_plan_approved:
        logging.info("Plan approved by user or auto-approved.") # This covers auto-approve path too if we reach here
    # Refinement and cancellation already logged when they occur and break the loop.
    # If it's an auto-approval, it returns earlier.

    logging.info("Exiting human_approval_node.")
    return {"plan_approved": current_plan_approved, "user_feedback": current_user_feedback}


def should_continue_decider(state: BuddyGraphState) -> str:
    logging.info("Entering new should_continue_decider.")
    objective = state["objective"]
    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)
    step_results = state.get("step_results", [])

    # Initial checks for critical errors or invalid plan
    if not plan or not isinstance(plan, list) or not plan[0]: # Check if plan is empty or first step is empty
        logging.warning("Decider: Invalid or empty plan provided.")
        # Try to replan if objective exists, otherwise critical error
        return "replan" if objective else "critical_error"

    if plan[0].startswith("Critical Error: Planner LLM not initialized.") or \
       plan[0].startswith("Critical Error: Planner failed to generate a plan.") or \
       plan[0].startswith("Critical Error: Planner returned non-string steps.") or \
       plan[0].startswith("Critical Error: Planner returned no steps or invalid plan structure."):
        logging.error(f"Decider: Critical error in plan generation: {plan[0]}")
        return "critical_error"

    last_step_result = step_results[-1] if step_results else ""
    if last_step_result.startswith("Critical Error: Executor agent not initialized.") or \
       last_step_result.startswith("Error: Invalid plan, step index out of bounds, or empty step content."): # Executor specific critical errors
        logging.error(f"Decider: Critical error during execution: {last_step_result}")
        return "critical_error"

    # Check if normal execution path has finished all steps
    if current_idx >= len(plan):
        logging.info("Decider: All steps executed. Assessing if objective is met.")

        global _assessor_llm_global
        if not _assessor_llm_global:
            logging.error("Decider: Assessor LLM not initialized. Cannot assess objective.")
            return "critical_error" # Cannot make a decision

        original_plan_str = "\n".join(f"- {s}" for s in plan)
        step_results_formatted = "\n".join(f"Step {i+1} Result: {res}" for i, res in enumerate(step_results))

        prompt = _ASSESSMENT_PROMPT_TEMPLATE.format(
            objective=objective,
            original_plan=original_plan_str,
            step_results_formatted=step_results_formatted
        )
        logging.debug(f"Assessment prompt: {prompt}")

        try:
            # Forcing a structured response for assessment
            assessor_llm_structured = _assessor_llm_global.with_structured_output(Assessment)
            ai_response = assessor_llm_structured.invoke(prompt)

            if not isinstance(ai_response, Assessment):
                logging.error(f"Decider: Assessment LLM returned unexpected type: {type(ai_response)}. Content: {ai_response}")
                # Fallback: assume replan is needed if assessment fails structurally
                return "replan"

            logging.info(f"Decider: Assessment received: Objective Met={ai_response.objective_met}, Reasoning='{ai_response.reasoning}', Suggest Replan={ai_response.suggest_replanning}")

            if ai_response.objective_met:
                logging.info("Decider: Objective MET.")
                return "objective_achieved"
            elif ai_response.suggest_replanning:
                logging.info("Decider: Objective NOT met, replanning suggested.")
                return "replan"
            else:
                logging.warning("Decider: Objective NOT met, replanning NOT suggested. Treating as error/end.")
                # This case might mean the task is impossible or failed critically according to LLM.
                return "critical_error" # Or a new state like "objective_failed_end"

        except Exception as e:
            logging.error(f"Decider: Error invoking assessment LLM: {e}", exc_info=True)
            # Fallback: if LLM call fails, assume replan is needed to be safe
            return "replan"

    # If plan is not exhausted and no critical errors encountered so far in this function
    logging.info(f"Decider: Continuing to execute step {current_idx + 1}/{len(plan)}.")
    return "continue_to_executor"

# --- Graph Setup ---
def set_global_llms_and_agents(planner_llm: ChatGoogleGenerativeAI, executor_agent: callable): # Modified signature
    global _planner_llm_structured, _executor_agent_runnable_global, _assessor_llm_global # Added assessor
    try:
        _planner_llm_structured = planner_llm.with_structured_output(Plan)
        logging.info("Global planner LLM configured for structured output (Plan).")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to configure global planner LLM for structured output: {e}", exc_info=True)
        raise
    _executor_agent_runnable_global = executor_agent
    _assessor_llm_global = planner_llm # Use the same base LLM for assessment
    logging.info("Global executor agent runnable and assessor LLM set.")

def create_buddy_graph() -> StateGraph:
    logging.info("Defining StateGraph workflow...")
    workflow = StateGraph(BuddyGraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("human_approval", human_approval_node) # New node

    workflow.set_entry_point("planner")

    # Planner to Human Approval
    workflow.add_edge("planner", "human_approval")

    # Human Approval conditional logic
    def decide_after_approval(state: BuddyGraphState) -> str:
        logging.info("Entering decide_after_approval.")
        plan_approved = state.get("plan_approved", False)
        user_feedback = state.get("user_feedback") # This will be the actual feedback string or None
        plan = state.get("plan", [])

        plan_has_critical_error = not plan or not isinstance(plan, list) or not plan[0] or plan[0].startswith("Critical Error:")
        logging.debug(f"State for decide_after_approval: plan_approved={plan_approved}, user_feedback_present={bool(user_feedback)}, plan_has_critical_error={plan_has_critical_error})")

        next_node_name = END # Default to END

        if plan_has_critical_error:
             logging.error(f"Critical error in plan detected by decide_after_approval: {plan[0] if plan and isinstance(plan, list) and plan[0] else 'Plan is empty or invalid'}")
             next_node_name = END
        elif plan_approved:
            # logging.info("Plan approved. Proceeding to executor.") # Logged before returning route
            next_node_name = "executor"
        elif user_feedback: # and not plan_approved (implicit from human_approval_node logic)
            # logging.info("User feedback provided. Proceeding to replanner.") # Logged before returning route
            next_node_name = "replanner"
        else: # Not approved, no feedback (e.g. user cancelled)
            # logging.info("Plan not approved and no feedback. Ending.") # Logged before returning route
            next_node_name = END

        logging.info(f"Routing from human_approval_node to: {'END' if next_node_name is END else next_node_name}")
        return next_node_name

    workflow.add_conditional_edges(
        "human_approval",
        decide_after_approval,
        {
            "executor": "executor",
            "replanner": "replanner",
            END: END
        }
    )

    # Executor to should_continue_decider
    workflow.add_conditional_edges(
        "executor",
        should_continue_decider, # This decider now primarily handles post-execution logic
        {
            "continue_to_executor": "executor", # This path should ideally not be hit if plan is single step. If plan has multiple steps, this is fine.
            "replan": "replanner", # If assessment suggests replan
            "objective_achieved": END,
            "critical_error": END # If assessment or execution had critical error
        }
    )

    # Replanner back to Human Approval
    workflow.add_edge("replanner", "human_approval")

    app = workflow.compile()
    logging.info("StateGraph compiled successfully.")
    return app

# --- Console Setter for Agent ---
def set_agent_console(console: Console):
    global _agent_cli_console
    _agent_cli_console = console
    logging.info("Agent CLI console set.")
