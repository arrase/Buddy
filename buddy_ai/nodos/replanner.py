import logging

from . import BuddyGraphState
from .planner import Plan
from .. import shared_instances as replanner_shared_instances # Alias for clarity
# from ..shared_instances import _planner_llm_structured


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


def replanner_node(state: BuddyGraphState) -> dict:
    logging.info("Entering replanner_node.")
    objective = state["objective"]
    previous_plan = state.get("plan", [])
    step_results = state.get("step_results", [])
    context = state.get("context", "")

    if not replanner_shared_instances._planner_llm_structured:
        logging.error("Replanner Node: Planner LLM (_planner_llm_structured from shared_instances) is not initialized.")
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
        ai_response = replanner_shared_instances._planner_llm_structured.invoke(formatted_prompt)
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
