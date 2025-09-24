# dream/cli.py
import runpy
import sys

def run():
    # this will execute dream/main.py as if you did `python -m dream.main`
    runpy.run_module("dream.main", run_name="__main__", alter_sys=True)
