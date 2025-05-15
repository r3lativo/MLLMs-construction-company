from pydantic import BaseModel, Field
from typing import Any
import logging

logger = logging.getLogger(__name__)


class GroupedActionOutput(BaseModel):
    """
    Represents the Builder's action output where actions are grouped by type.
    Matches the target dictionary structure:
    {
        "actions": {
            "action_type_1": [
                [param1_val1, param2_val1, ...], # parameters for instance 1 of type 1
                [param1_val2, param2_val2, ...], # parameters for instance 2 of type 1
                ...
            ],
            "action_type_2": [
                [param1_valA, ...], # parameters for instance 1 of type 2
                ...
            ]
        },
        "communication": "..."
    }
    """
    # The 'actions' field is a dictionary:
    # - Keys are strings (the action types like "place_block")
    # - Values are lists of lists (each inner list is the params for one action instance)
    actions: dict[str, list[list[Any]]] = Field(
        default_factory=dict,
        description="Dictionary where keys are action types and values are lists of parameter lists."
    )

    # The 'communication' field is similar to the old model
    communication: str = Field(
        default="",
        description="Optional message to the Architect after performing actions."
    )
