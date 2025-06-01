from typing import Optional, Callable
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console

# These variables will be initialized by functions in agent.py
_planner_llm_structured: Optional[ChatGoogleGenerativeAI] = None
_executor_agent_runnable_global: Optional[Callable] = None
_assessor_llm_global: Optional[ChatGoogleGenerativeAI] = None
_agent_cli_console: Optional[Console] = None
