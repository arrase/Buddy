import logging
from pydantic import BaseModel, Field
from langgraph.graph import END

# Import BuddyGraphState from the __init__.py in the same directory
from . import BuddyGraphState
# Import the global variable from the original agent module
from ..agent import _assessor_llm_global


class Assessment(BaseModel):
    objective_met: bool = Field(description="Whether the user's objective has been fully met.")
    reasoning: str = Field(description="Brief explanation for why the objective is considered met or not.")
    suggest_replanning: bool = Field(description="If objective is not met, should we try to replan? Set to false if errors are unrecoverable or further attempts seem futile.")


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


def should_continue_decider(state: BuddyGraphState) -> str:
    logging.info("Entering new should_continue_decider.")
    objective = state["objective"]
    plan = state.get("plan")
    current_idx = state.get("current_step_index", 0)
    step_results = state.get("step_results", [])

    if not plan or not isinstance(plan, list) or not plan[0]:
        logging.warning("Decider: Invalid or empty plan provided.")
        return "replan" if objective else "critical_error" # If no objective, cannot replan

    # Check for critical errors from planner
    if plan[0].startswith("Critical Error: Planner LLM not initialized.") or \
       plan[0].startswith("Critical Error: Planner failed to generate a plan.") or \
       plan[0].startswith("Critical Error: Planner returned non-string steps.") or \
       plan[0].startswith("Critical Error: Planner returned no steps or invalid plan structure."):
        logging.error(f"Decider: Critical error in plan generation: {plan[0]}")
        return "critical_error"

    # Check for critical errors from executor (if any steps have run)
    last_step_result = step_results[-1] if step_results else ""
    if last_step_result.startswith("Critical Error: Executor agent not initialized.") or \
       last_step_result.startswith("Error: Invalid plan, step index out of bounds, or empty step content."): # This error is from executor
        logging.error(f"Decider: Critical error during execution: {last_step_result}")
        return "critical_error"

    if current_idx >= len(plan):
        logging.info("Decider: All steps executed. Assessing if objective is met.")

        global _assessor_llm_global
        if not _assessor_llm_global:
            logging.error("Decider: Assessor LLM not initialized. Cannot assess objective.")
            return "critical_error" # Cannot proceed without assessor

        original_plan_str = "\n".join(f"- {s}" for s in plan)
        step_results_formatted = "\n".join(f"Step {i+1} Result: {res}" for i, res in enumerate(step_results))

        prompt = _ASSESSMENT_PROMPT_TEMPLATE.format(
            objective=objective,
            original_plan=original_plan_str,
            step_results_formatted=step_results_formatted
        )
        logging.debug(f"Assessment prompt: {prompt}")

        try:
            # Assuming _assessor_llm_global is already configured for structured_output(Assessment)
            # This was done in set_global_llms_and_agents
            ai_response = _assessor_llm_global.with_structured_output(Assessment).invoke(prompt)


            if not isinstance(ai_response, Assessment):
                logging.error(f"Decider: Assessment LLM returned unexpected type: {type(ai_response)}. Content: {ai_response}")
                # If LLM fails to structure, consider it a need to replan, as assessment is unclear.
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
                # If objective not met and replan not suggested, it's a terminal state.
                return "critical_error"

        except Exception as e:
            logging.error(f"Decider: Error invoking assessment LLM: {e}", exc_info=True)
            # If assessment fails, safest is to try replanning.
            return "replan"

    logging.info(f"Decider: Continuing to execute step {current_idx + 1}/{len(plan)}.")
    return "continue_to_executor"


def decide_after_approval(state: BuddyGraphState) -> str:
    logging.info("Entering decide_after_approval.")
    plan_approved = state.get("plan_approved", False)
    user_feedback = state.get("user_feedback")
    plan = state.get("plan", [])

    # Check if the plan itself contains a critical error message (e.g. from human_approval_node console error)
    plan_has_critical_error = not plan or not isinstance(plan, list) or not plan[0] or plan[0].startswith("Critical Error:")

    logging.debug(f"State for decide_after_approval: plan_approved={plan_approved}, user_feedback_present={bool(user_feedback)}, plan_has_critical_error={plan_has_critical_error}")

    next_node_name = END # Default to END if no other condition met

    if plan_has_critical_error:
         logging.error(f"Critical error in plan detected by decide_after_approval: {plan[0] if plan and isinstance(plan, list) and plan[0] else 'Plan is empty or invalid'}")
         next_node_name = END # Critical error in plan, should end.
    elif plan_approved:
        next_node_name = "executor"
    elif user_feedback: # Implies plan was not approved and user wants to refine
        next_node_name = "replanner"
    else: # Plan not approved, no feedback (e.g., user cancelled, or an issue in human_approval_node)
        next_node_name = END

    logging.info(f"Routing from human_approval_node to: {'END' if next_node_name is END else next_node_name}")
    return next_node_name
