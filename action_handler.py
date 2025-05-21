import logging
import json

class World:
    """
    Represents the virtual 3D world where blocks are placed and managed.
    It provides methods to add, remove, and query blocks, and to represent
    the world state in various formats.
    """
    def __init__(self):
        """
        Initializes the World with an empty set of blocks.
        Blocks are stored in a dictionary where keys are (x, y, z) tuples
        and values are their corresponding colors.
        """
        self.blocks = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("World initialized: Ready to manage blocks.")

    def place_block(self, color: str, x: any, y: any, z: any) -> bool:
        """
        Places a colored block at the specified coordinates in the world.
        If a block already exists at the coordinates, it will be replaced.

        Args:
            color (str): The color of the block to place (e.g., "red", "blue").
            x (any): The x-coordinate for the block. Will be converted to int.
            y (any): The y-coordinate for the block. Will be converted to int.
            z (any): The z-coordinate for the block. Will be converted to int.

        Returns:
            bool: True if the block was successfully placed, False otherwise.
        """
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            # Log an error for invalid coordinates without being too verbose
            self.logger.error(f"Failed to place block: Invalid coordinates received ({x}, {y}, {z}).")
            return False

        coords = (x, y, z)
        if coords in self.blocks:
            self.logger.info(f"Replacing existing block at {coords} with {color}.")
        else:
            self.logger.info(f"Placing {color} block at {coords}.")
            
        self.blocks[coords] = color
        return True

    def remove_block(self, x: any, y: any, z: any) -> bool:
        """
        Removes a block from the specified coordinates in the world.

        Args:
            x (any): The x-coordinate of the block to remove. Will be converted to int.
            y (any): The y-coordinate of the block to remove. Will be converted to int.
            z (any): The z-coordinate of the block to remove. Will be converted to int.

        Returns:
            bool: True if a block was successfully removed, False if no block was found.
        """
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            self.logger.error(f"Failed to remove block: Invalid coordinates received ({x}, {y}, {z}).")
            return False

        coords = (x, y, z)
        if coords in self.blocks:
            del self.blocks[coords]
            self.logger.info(f"Removed block at {coords}.")
            return True
        else:
            # Log as info if no block was found, as it might be an expected scenario
            self.logger.info(f"No block found at {coords} to remove.")
            return False

    def get_block(self, x: int, y: int, z: int):
        """
        Retrieves the color of the block at the specified coordinates.

        Args:
            x (int): The x-coordinate.
            y (int): The y-coordinate.
            z (int): The z-coordinate.

        Returns:
            str | None: The color of the block if found, otherwise None.
        """
        try:
            x, y, z = int(x), int(y), int(z)
        except ValueError:
            self.logger.error(f"Invalid coordinates for get_block: ({x}, {y}, {z}).")
            return None

        return self.blocks.get((x, y, z))

    def get_state_as_json(self) -> list[dict]:
        """
        Returns the current state of all blocks in the world as a list of dictionaries.
        Each dictionary represents a block with its 'block_color', 'x', 'y', and 'z' coordinates.

        Returns:
            list[dict]: A JSON-serializable list representing the world state.
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
        Generates a concise textual description of the current world state, primarily for the Architect agent.
        If the world is empty, it returns a specific message. Otherwise, it provides a
        pretty-printed JSON representation of all placed blocks.

        Returns:
            str: A string describing the current state of the world.
        """
        if not self.blocks:
            return "The world is currently empty."
        else:
            data = self.get_state_as_json()
            world_state_description = json.dumps(data, indent=2)
            self.logger.debug("Generated world state description for Architect.")
            return f"Current World State:\n{world_state_description}"

    def __str__(self) -> str:
        """
        Provides a string representation of the World, which is the detailed
        world state description intended for the Architect.
        """
        return self.get_state_description_for_architect()
