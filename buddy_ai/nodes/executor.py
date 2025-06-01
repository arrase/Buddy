import logging
from langchain_core.messages import HumanMessage

# Import BuddyGraphState from the __init__.py in the same directory
from . import BuddyGraphState
from .. import shared_instances as executor_shared_instances # Alias for clarity


def executor_node(state: BuddyGraphState) -> dict:
    current_idx = state.get("current_step_index", 0)
    logging.info(f"Entering executor_node for step index {current_idx}.")
    logging.info(f"EXECUTOR: shared_instances module ID: {id(executor_shared_instances)}")
    logging.info(f"EXECUTOR: _executor_agent_runnable_global ID from shared_instances: {id(executor_shared_instances._executor_agent_runnable_global)}")

    if not executor_shared_instances._executor_agent_runnable_global:
        logging.error("Executor agent (_executor_agent_runnable_global from shared_instances) is not initialized.")
        return {"step_results": ["Critical Error: Executor agent not initialized."], "current_step_index": current_idx + 1}

    plan = state.get("plan")
    step_output_str = "Error: Pre-execution state error in executor_node."
    next_step_idx = current_idx + 1

    if not plan or not isinstance(plan, list) or not (0 <= current_idx < len(plan)) or not plan[current_idx]:
        step_output_str = "Error: Invalid plan, step index out of bounds, or empty step content."
        logging.error(step_output_str)
        if plan and isinstance(plan, list) and len(plan) > 0 and plan[0].startswith("Critical Error"):
            step_output_str = plan[0] # Propagate critical planner errors
    else:
        current_instruction = plan[current_idx]
        logging.info(f"Executing step {current_idx + 1}/{len(plan)}: {current_instruction}")
        try:
            agent_input = {"messages": [HumanMessage(content=current_instruction)]}
            agent_response = executor_shared_instances._executor_agent_runnable_global.invoke(agent_input)
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
