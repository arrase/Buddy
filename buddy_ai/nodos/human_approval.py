import logging
from rich.markdown import Markdown

# Import BuddyGraphState from the __init__.py in the same directory
from . import BuddyGraphState
from ..shared_instances import _agent_cli_console


from .. import shared_instances as ha_shared_instances # Use an alias for clarity

def human_approval_node(state: BuddyGraphState) -> dict:
    logging.info("Entering human_approval_node.")
    logging.info(f"HUMAN_APPROVAL: shared_instances module ID: {id(ha_shared_instances)}")
    logging.info(f"HUMAN_APPROVAL: _agent_cli_console ID from shared_instances: {id(ha_shared_instances._agent_cli_console)}")


    if ha_shared_instances._agent_cli_console is None:
        logging.critical("CRITICAL: Agent CLI console (_agent_cli_console from shared_instances) not set in human_approval_node. This should have been set by the CLI.")
        # This state indicates a severe setup error. The plan list itself is modified to reflect this.
        return {"plan_approved": False, "user_feedback": "Critical Error: Agent console not configured.", "plan": ["Critical Error: Agent console not configured."]}

    auto_approve = state.get("auto_approve", False)
    logging.info(f"Plan approval state: auto_approve={auto_approve}, current_plan_approved={state.get('plan_approved')}, user_feedback_present={bool(state.get('user_feedback'))}")

    current_plan_approved = False  # Default to not approved
    current_user_feedback = None

    if auto_approve:
        logging.info("Auto-approving plan.")
        current_plan_approved = True
        # Return immediately if auto-approving
        return {"plan_approved": current_plan_approved, "user_feedback": current_user_feedback}

    plan = state.get("plan")

    # Check if the plan itself indicates a critical error from a previous node (e.g., planner)
    if not plan or not isinstance(plan, list) or not plan[0] or plan[0].startswith("Critical Error:"):
        logging.warning(f"Critical error in plan detected by human_approval_node. Bypassing user interaction. Plan: {plan}")
        # No approval, no feedback, pass the potentially error-containing plan through.
        return {"plan_approved": False, "user_feedback": None, "plan": plan}

    plan_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    _agent_cli_console.print(Markdown(f"## Proposed Execution Plan:\n{plan_md}"))

    while True:  # Loop until valid input (A, R, C) is received
        try:
            raw_input = _agent_cli_console.input("[bold yellow]Plan Review[/bold yellow]: ([bold green]A[/bold green])pprove, ([bold blue]R[/bold blue])efine, or ([bold red]C[/bold red])ancel plan? ").strip().lower()
        except KeyboardInterrupt: # Treat Ctrl+C as cancel
            logging.warning("User cancelled via KeyboardInterrupt during plan approval.")
            raw_input = 'c' # Simulate cancel input

        if raw_input == 'a':
            logging.info("Plan approved by user.")
            current_plan_approved = True
            current_user_feedback = None
            break
        elif raw_input == 'r':
            logging.info("User chose to refine the plan.")
            while True: # Loop for getting refinement feedback
                try:
                    feedback = _agent_cli_console.input("Please provide feedback for replanning: ").strip()
                    if feedback:
                        current_user_feedback = feedback
                        current_plan_approved = False # Plan is not approved if feedback is given
                        break
                    else:
                        _agent_cli_console.print("[bold red]Feedback cannot be empty if you choose to refine. Please provide your comments or (C)ancel refinement.[/bold red]")
                        # Allow user to cancel out of refinement feedback loop
                        sub_choice = _agent_cli_console.input("Enter feedback or (C)ancel refinement: ").strip().lower()
                        if sub_choice == 'c':
                            current_user_feedback = None # Ensure feedback is None if refinement is cancelled
                            break # Exits refinement feedback loop, re-prompts A/R/C
                except KeyboardInterrupt:
                    logging.warning("User cancelled refinement input via KeyboardInterrupt.")
                    current_user_feedback = None # Ensure feedback is None
                    break # Exits refinement feedback loop
            if current_user_feedback is not None: # If feedback was successfully provided
                break # Exits main A/R/C loop
            # If refinement was cancelled (current_user_feedback is None), continue main A/R/C loop

        elif raw_input == 'c':
            current_plan_approved = False
            current_user_feedback = None
            logging.info("User cancelled plan approval.")
            break
        else:
            logging.debug("Invalid input from user during plan approval.")
            _agent_cli_console.print("[bold red]Invalid input. Please enter 'A', 'R', or 'C'.[/bold red]")

    # Log final decision from this node
    if current_plan_approved:
        logging.info("Plan approved by user or auto-approved.")
    elif current_user_feedback:
        logging.info(f"Plan refinement requested with feedback: {current_user_feedback}")
    else:
        logging.info("Plan explicitly cancelled by user or due to error.")

    logging.info("Exiting human_approval_node.")
    return {"plan_approved": current_plan_approved, "user_feedback": current_user_feedback}
