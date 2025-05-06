import yaml
import os

def load_config(config_path="config/config.yaml"):
    """Loads configuration from the specified YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return None

# Load it directly when the module is imported
# Just call it with `config.get("<KEY>")`
config = load_config()