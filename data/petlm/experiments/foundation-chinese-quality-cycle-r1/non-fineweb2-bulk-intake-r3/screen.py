"""Run the R3 review pipeline with the historical intake layout."""
from pathlib import Path
from runpy import run_path


if __name__ == '__main__':
    run_path(str(Path(__file__).with_name('expand.py')), run_name='__main__')
