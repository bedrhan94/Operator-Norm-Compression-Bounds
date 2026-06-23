"""Make the repository root importable so tests can ``from src... import ...``."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
