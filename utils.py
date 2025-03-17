# utils.py

import json

def load_persona():
    """Load persona data from JSON file."""
    with open("persona_data.json", "r") as f:
        return json.load(f)
