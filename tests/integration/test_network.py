import shutil
from pathlib import Path
import pytest


from legal_system_rag.network.client_factory import (
    check_if_px_is_running,
    create_ssl_context,
    create_http_client,
    create_async_http_client,
    get_proxy_url,
)


# Prüfen, ob px auf dem System installiert ist
PX_PATH = Path(r"C:\Program Files\px\px.exe")
HAS_PX_INSTALLED = PX_PATH.exists() or shutil.which("px") is not None


@pytest.mark.proxy
@pytest.mark.skipif(not HAS_PX_INSTALLED, reason="Px ist auf diesem System nicht installiert")
def test_px():
    """Prüft, ob der Px-Proxy läuft (wird durch conftest.py gestartet)."""
    assert check_if_px_is_running() is True


def test_ssl_default():
    """Prüft die Standard-SSL-Konfiguration (Zertifikatsprüfung aktiv)."""
    ssl_context = create_ssl_context(ignore_ssl=False)

    assert ssl_context.verify_mode == 2  # ssl.CERT_REQUIRED
    assert ssl_context.check_hostname is True


def test_ssl_ignore():
    """Prüft das Deaktivieren der SSL-Verifikation."""
    ssl_context = create_ssl_context(ignore_ssl=True)

    assert ssl_context.verify_mode == 0  # ssl.CERT_NONE
    assert ssl_context.check_hostname is False

def test_network_client():
    """Synchronous HTTP client test through the proxy."""
    proxy_url = get_proxy_url()

    with create_http_client(proxy_url=proxy_url) as client:
        response = client.get("https://github.com", timeout=10)

    assert response.status_code == 200

@pytest.mark.anyio
async def test_async_network_client():
    """Asynchronous HTTP client test through the proxy."""
    proxy_url = get_proxy_url()

    async with create_async_http_client(proxy_url=proxy_url) as client:
        response = await client.get("https://github.com", timeout=10)

    assert response.status_code == 200

        