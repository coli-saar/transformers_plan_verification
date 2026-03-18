from pathlib import Path


PROJ_DIR = Path(__file__).parent.parent
TEMP_DIR = PROJ_DIR / Path('temp_files')
VAL = PROJ_DIR / 'planning_utils/VAL'

def create_temp_dir():
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
