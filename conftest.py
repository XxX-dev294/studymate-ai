"""Гарантирует, что корень проекта в sys.path при запуске `pytest` из
любой директории (а не только через `python -m pytest`)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
