from typing import TypedDict, List, Optional, Annotated
import operator

class BuddyGraphState(TypedDict):
    objective: str
    context: Optional[str]
    plan: Optional[List[str]]
    current_step_index: int
    step_results: Annotated[List[str], operator.add]
    final_output: Optional[str]
    user_feedback: Optional[str] = None
    plan_approved: bool = False
    auto_approve: bool = False
