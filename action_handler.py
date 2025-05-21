import logging
import json

class World:
    """Represents the virtual world where blocks are placed."""
    def __init__(self):
        self.blocks = {} # Stores blocks as {(x, y, z): color}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("World initialized.")

    def place_block(self, color: str, x: any, y: any, z: any):
        """Places a block at the specified coordinates with a color."""
        # Basic type conversion for parameters received from the model
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            self.logger.error(f"Error: Invalid coordinates for place_block: ({x}, {y}, {z})")
            return False

        coords = (x, y, z)
        if coords in self.blocks:
            self.logger.warning(f"World: Replacing existing block at {coords}")
        self.blocks[coords] = color # Store block color
        self.logger.info(f"World: Placed {color} block at {coords}")
        return True # Indicate success

    def remove_block(self, x: any, y: any, z: any):
        """Removes a block at the specified coordinates."""
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            self.logger.error(f"Error: Invalid coordinates for remove_block: ({x}, {y}, {z})")
            return False

        coords = (x, y, z)
        if coords in self.blocks:
            del self.blocks[coords]
            self.logger.info(f"World: Removed block at {coords}")
            return True
        else:
            self.logger.warning(f"World: No block found at {coords} to remove.")
            return False

    def get_block(self, x: int, y: int, z: int):
        """Gets the block color at the specified coordinates."""
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            self.logger.error(f"Error: Invalid coordinates for get_block: ({x}, {y}, {z})")
            return None # Invalid coordinates

        return self.blocks.get((x, y, z))

    def get_state_as_json(self) -> list[dict]:
        """
        Returns the world state as a JSON-serializable list of dictionaries,
        matching the structure: [{"block_color": "...", "x": ..., "y": ..., "z": ...}].
        """
        world_blocks_list = []
        for (x, y, z), color in self.blocks.items():
            world_blocks_list.append({
                "block_color": color,
                "x": x,
                "y": y,
                "z": z
            })
        return world_blocks_list

    def get_state_description_for_architect(self) -> str:
        """
        Generates a textual description of the current world state for the Architect.
        Returns "The world is currently empty." if no blocks, otherwise returns
        the JSON representation of the world state as a formatted string.
        """
        if not self.blocks:
            return "The world is currently empty."
        else:
            # Get the list of dictionaries and convert it to a pretty-printed JSON string
            data = self.get_state_as_json()
            world_state_description = json.dumps(data, indent=2)
            return f"Current World State:\n{world_state_description}"

    def __str__(self):
        """Returns the world state description for the Architect."""
        return self.get_state_description_for_architect()
