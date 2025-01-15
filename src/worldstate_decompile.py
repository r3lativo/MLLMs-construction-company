import json
from typing import List, Dict, Optional  # Importing type hints for better clarity and type checking
from dataclasses import dataclass  # Importing dataclass to simplify the creation of classes


# Define the classes to represent the data structure in the JSON

@dataclass
class BuilderPosition:
    # Represents the position and orientation of the builder
    Y: float  # Vertical position (height)
    X: float  # Horizontal position (east-west)
    Yaw: float  # Rotation around the vertical axis
    Z: float  # Depth position (north-south)
    Pitch: float  # Rotation around the horizontal axis


@dataclass
class BuilderInventoryItem:
    # Represents an item in the builder's inventory
    Index: int  # Index in the inventory list
    Type: str  # Type of the item (e.g., block type)
    Quantity: int  # Number of items of this type


@dataclass
class Screenshots:
    # Represents the screenshots associated with a session
    FixedViewer4: Optional[str]  # Screenshot from fixed viewer 4, if available
    FixedViewer1: Optional[str]  # Screenshot from fixed viewer 1, if available
    FixedViewer2: Optional[str]  # Screenshot from fixed viewer 2, if available
    FixedViewer3: Optional[str]  # Screenshot from fixed viewer 3, if available
    Builder: Optional[str]  # Screenshot from the builder's perspective
    Architect: Optional[str]  # Screenshot from the architect's perspective


@dataclass
class WorldState:
    # Represents the state of the world at a particular moment
    BuilderPosition: BuilderPosition  # The builder's position and orientation
    ChatHistory: List[str]  # List of chat messages exchanged
    Timestamp: str  # Timestamp of this world state
    BlocksInGrid: List[Dict]  # List of blocks in the grid (e.g., positions, types)
    BuilderInventory: List[BuilderInventoryItem]  # Items in the builder's inventory
    Screenshots: Screenshots  # Associated screenshots


@dataclass
class WorldStateData:
    # Represents the entire dataset containing multiple world states
    WorldStates: List[WorldState]  # List of world state objects

    @staticmethod
    def from_dict(data: dict) -> 'WorldStateData':
        """
        Static method to convert a dictionary (e.g., parsed JSON) 
        into a WorldStateData object containing structured data.
        """
        world_states = [  # Loop through the list of world states in the dictionary
            WorldState(
                BuilderPosition=BuilderPosition(**ws['BuilderPosition']) if ws['BuilderPosition'] is not None else [],  # Safely unpack or provide a default
                ChatHistory=ws['ChatHistory'],  # Extract chat history as-is
                Timestamp=ws['Timestamp'],  # Extract timestamp as-is
                BlocksInGrid=ws['BlocksInGrid'],  # Extract blocks in the grid as-is
                BuilderInventory=[
                    BuilderInventoryItem(**item) for item in ws['BuilderInventory']
                ] if ws['BuilderInventory'] is not None else [],  # Handle missing inventory data
                Screenshots=Screenshots(**ws['Screenshots'] if ws['Screenshots'] else {})  # Handle missing screenshots
            )
            for ws in data.get('WorldStates', [])  # Iterate over all world states
        ]
        
        # Return the structured WorldStateData object
        return WorldStateData(WorldStates=world_states)
