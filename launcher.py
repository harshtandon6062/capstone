"""Application entry point. The launcher is now just the orchestrator."""

import os


def run_launcher():
    from main import run_pick_and_place

    print("Launching gesture-controlled pick-and-place application.")
    run_pick_and_place()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_launcher()
import sys
