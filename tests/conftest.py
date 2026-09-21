
               
# tests/conftest.py
import shutil
import subprocess
import time
from pathlib import Path

import pytest

# Direkt aus deinem Modul importieren, nicht neu schreiben!
from legal_system_rag.network.client_factory import check_if_px_is_running

PX_PATH = Path(r"C:\Program Files\px\px.exe")
HAS_PX_INSTALLED = PX_PATH.exists() or shutil.which("px") is not None


def wait_for_px_ready(timeout: float = 10.0, poll_interval: float = 0.5) -> bool:
    """
    Pollt check_if_px_is_running() bis Px ready ist oder Timeout eintritt.
    
    Returns:
        True wenn Px innerhalb des Timeouts ready wurde, sonst False.
    """
    start = time.perf_counter()
    while (time.perf_counter() - start) < timeout:
        if check_if_px_is_running():
            return True
        time.sleep(poll_interval)
    return False


@pytest.fixture(scope="session", autouse=True)
def ensure_px_proxy_is_running():
    """Startet Px vor allen Tests und wartet bis es wirklich ready ist."""
    if not HAS_PX_INSTALLED:
        yield None
        return

    process = None
    
    # Falls Px schon läuft (vom vorherigen Testlauf), nicht neu starten
    if not check_if_px_is_running():
        cmd = str(PX_PATH) if PX_PATH.exists() else "px"
        process = subprocess.Popen(
            [cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        
        # ✅ Robuster Health-Check statt fixem sleep()
        if not wait_for_px_ready(timeout=15.0, poll_interval=0.5):
            process.terminate()
            process.wait()
            pytest.fail(
                "Px-Proxy startete nicht innerhalb von 15 Sekunden. "
                "Prüfe px.ini Konfiguration und Proxy-Einstellungen."
            )
    # else: Px läuft bereits (z.B. vom vorherigen pytest-Lauf)

    yield process

    if process is not None:
        process.terminate()       
        process.wait()

        