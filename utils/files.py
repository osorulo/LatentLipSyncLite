from pathlib import Path
import shutil
import uuid

WORKDIR = Path("gradio_tmp")
WORKDIR.mkdir(exist_ok=True)

def create_run_dir():
    run_id = uuid.uuid4().hex[:8]
    run_dir = WORKDIR / run_id
    run_dir.mkdir(exist_ok=True)
    return run_id, run_dir, WORKDIR

def cleanup_dir(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
