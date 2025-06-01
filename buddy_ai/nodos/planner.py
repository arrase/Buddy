import logging
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from . import BuddyGraphState
# Import the global variable from the original agent module
from ..agent import _planner_llm_structured


class Plan(BaseModel):
    steps: List[str] = Field(description="Actionable steps for the executor to follow.")


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


def planner_node(state: BuddyGraphState) -> dict:
    logging.info("Entering planner_node.")
    objective = state["objective"]
    context = state.get("context", "")

    # Access the globally defined _planner_llm_structured
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
