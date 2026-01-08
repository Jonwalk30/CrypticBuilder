import yaml
from pathlib import Path

class ScoringConfig:
    def __init__(self, config_path: str = "ConfigurationMetrics.yml"):
        try:
            with open(config_path, "r") as f:
                self.data = yaml.safe_load(f)
        except FileNotFoundError:
            self.data = {}

    def get(self, section: str):
        return self.data.get(section, {})

# Global instance for easy access
config = ScoringConfig()
