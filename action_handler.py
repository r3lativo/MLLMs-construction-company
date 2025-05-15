import logging

logger = logging.getLogger(__name__)


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

    def get_state_as_json(self):
        """Returns the world state as a JSON-serializable dictionary."""
        # Convert tuple keys to strings for JSON compatibility
        return {f"{x},{y},{z}": color for (x, y, z), color in self.blocks.items()}

    def get_state_as_xml(self):
         """Returns the world state as a simple XML string (example)."""
         xml_string = "<world>\n"
         for (x, y, z), color in self.blocks.items():
              xml_string += f'  <block x="{x}" y="{y}" z="{z}" color="{color}"/>\n'
         xml_string += "</world>"
         return xml_string

    def get_state_description_for_architect(self):
        """Generates a textual description of the current world state for the Architect."""
        if not self.blocks:
            return "The world is currently empty."

        description = "Current blocks in the world:\n"
        # Limit the description length for context windows if needed
        items = []
        for (x, y, z), color in self.blocks.items():
            items.append(f"- {color} block at ({x}, {y}, {z})")
        # Sort for consistent output (optional)
        # items.sort()
        description += "\n".join(items[:20]) # List up to 20 blocks
        if len(items) > 20:
             description += f"\n... and {len(items) - 20} more blocks."
        return description

    def __str__(self):
        return self.get_state_description_for_architect()
