"""Put the repository root on the import path for the test run.

The tests live in tests/ but import the packages at the root (robot, workspace,
gestures, ui, commands) the same way the application does. Without this, pytest
would only put tests/ itself on sys.path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
