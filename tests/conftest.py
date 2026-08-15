import shutil
import subprocess
import time
from pathlib import Path

import pytest

# Direkt aus deinem Modul importieren, nicht neu schreiben!
from legal_system_rag.network.client_factory import check_if_px_is_running

PX_PATH = Path(r"C:\Program Files\px\px.exe")
HAS_PX_INSTALLED = PX_PATH.exists() or shutil.which("px") is not None


@pytest.fixture(scope="session", autouse=True)
def ensure_px_proxy_is_running():
    """Startet Px vor allen Tests, falls installiert aber inaktiv."""
    if not HAS_PX_INSTALLED:
        yield None
        return

    process = None
    if not check_if_px_is_running():
        cmd = str(PX_PATH) if PX_PATH.exists() else "px"
        process = subprocess.Popen(
            [cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3.0)

    yield process

    if process is not None:
        process.terminate()
        process.wait()