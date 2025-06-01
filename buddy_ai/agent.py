import logging
import operator
from typing import List, Optional, Callable # Removed TypedDict, Annotated as BuddyGraphState is imported
# from typing import TypedDict, List, Optional, Annotated # Keep for globals if necessary, or BuddyGraphState parts

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ShellTool
from langgraph.graph import StateGraph, END
# from pydantic import BaseModel, Field # BaseModel and Field are no longer used directly here
from langchain_core.messages import HumanMessage # Still needed for create_executor_agent_runnable
from langgraph.prebuilt import create_react_agent
# from rich.markdown import Markdown # Markdown is no longer used directly here
from rich.console import Console # Still needed for _agent_cli_console and set_agent_console

# Import the BuddyGraphState from the nodes package
from .nodes import BuddyGraphState

# Import node functions and relevant data classes from the nodes package
from .nodes.planner import planner_node, Plan
from .nodes.executor import executor_node
from .nodes.replanner import replanner_node
from .nodes.human_approval import human_approval_node
from .nodes.deciders import should_continue_decider, decide_after_approval
# Assessment is defined and used within deciders.py, not directly needed in agent.py's global scope.

# Global-like variables are now managed in shared_instances.py
# No global declarations here anymore.
from . import shared_instances


# Functions that are NOT nodes remain in agent.py
def create_llm_instance(model_name: str, llm_type: str, api_key: Optional[str] = None) -> Optional[ChatGoogleGenerativeAI]:
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        logging.info(f"{llm_type} LLM created successfully using {model_name}.")
        return llm
    except Exception as e:
        logging.error(f"Error creating {llm_type} LLM with {model_name}: {e}", exc_info=True)
        return None


def create_executor_agent_runnable(llm: ChatGoogleGenerativeAI) -> Optional[Callable]:
    try:
        # HumanMessage is used by the ReAct agent, so it's needed here.
        executor_agent = create_react_agent(llm, tools=[ShellTool()])
        logging.info("Executor ReAct agent created successfully with ShellTool.")
        return executor_agent
    except Exception as e:
        logging.error(f"Error creating Executor Agent: {e}", exc_info=True)
        return None


def set_global_llms_and_agents(planner_llm: ChatGoogleGenerativeAI, executor_agent: Callable):
    # from . import shared_instances # Already imported at module level
    try:
        # Plan is now imported from .nodes.planner
        logging.info(f"AGENT (set_globals): shared_instances module ID: {id(shared_instances)}")
        shared_instances._planner_llm_structured = planner_llm.with_structured_output(Plan)
        logging.info(f"AGENT (set_globals): _planner_llm_structured ID: {id(shared_instances._planner_llm_structured)}")
        logging.info("Global planner LLM configured for structured output (Plan) in shared_instances.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to configure global planner LLM for structured output: {e}", exc_info=True)
        raise
    shared_instances._executor_agent_runnable_global = executor_agent
    logging.info(f"AGENT (set_globals): _executor_agent_runnable_global ID: {id(shared_instances._executor_agent_runnable_global)}")
    # _assessor_llm_global is the same base LLM as planner, used by should_continue_decider.
    # should_continue_decider (in deciders.py) will call .with_structured_output(Assessment) on it.
    shared_instances._assessor_llm_global = planner_llm
    logging.info(f"AGENT (set_globals): _assessor_llm_global ID: {id(shared_instances._assessor_llm_global)}")
    logging.info("Global executor agent runnable and assessor LLM set in shared_instances.")


def create_buddy_graph() -> StateGraph:
    logging.info("Defining StateGraph workflow...")
    # BuddyGraphState is now imported
    workflow = StateGraph(BuddyGraphState)

    # Node functions are now imported
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("human_approval", human_approval_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "human_approval")

    # Conditional edge functions are now imported
    workflow.add_conditional_edges(
        "human_approval",
        decide_after_approval,
        {
            "executor": "executor",
            "replanner": "replanner",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "executor",
        should_continue_decider,
        {
            "continue_to_executor": "executor",
            "replan": "replanner",
            "objective_achieved": END,
            "critical_error": END
        }
    )

    workflow.add_edge("replanner", "human_approval")

    app = workflow.compile()
    logging.info("StateGraph compiled successfully.")
    return app


def set_agent_console(console: Console):
    # from . import shared_instances # Already imported at module level
    logging.info(f"AGENT: shared_instances module ID: {id(shared_instances)}")
    shared_instances._agent_cli_console = console
    logging.info(f"AGENT: Setting console in shared_instances: {id(shared_instances._agent_cli_console)}")
    logging.info("Agent CLI console set in shared_instances.")
