import logging
import socket
import ssl

import httpx

from legal_system_rag.config import (
    CERT_FILE,
    COMPANY_PROXY_URL,
    PX_HOST,
    PX_PORT,
    RETRIES,
    TIMEOUT,
)

logger = logging.getLogger(__name__)


def check_if_px_is_running() -> bool:
    """
    Prüft, ob der lokale PX-Proxy erreichbar ist.
    """
    try:
        with socket.create_connection(
            (PX_HOST, PX_PORT),
            timeout=1.0,
        ):
            return True

    except OSError:
        return False


def create_ssl_context(
    ignore_ssl: bool = False,
) -> ssl.SSLContext:
    """
    Erstellt den SSL-Kontext mit dem konfigurierten CA-Bundle.

    Args:
        ignore_ssl:
            Deaktiviert die Zertifikats- und Hostnamenprüfung.
            Nur für Tests verwenden.

    Raises:
        FileNotFoundError:
            Wenn CERT_FILE nicht existiert.
    """
    if not CERT_FILE.exists():
        raise FileNotFoundError(
            f"CA-Zertifikat wurde nicht gefunden: {CERT_FILE}"
        )

    context = ssl.create_default_context(
        cafile=str(CERT_FILE)
    )

    if ignore_ssl:
        logger.warning(
            "SSL-Zertifikatsprüfung ist deaktiviert. "
            "Nur für Tests verwenden!"
        )

        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    return context


def get_proxy_url() -> str | None:
    """
    Ermittelt den zu verwendenden Proxy.

    Priorität:
        1. Lokaler PX-Proxy.
        2. Unternehmensproxy.
        3. Direkte Verbindung.
    """
    if check_if_px_is_running():
        proxy_url = f"http://{PX_HOST}:{PX_PORT}"

        logger.debug(
            "Lokaler PX-Proxy wird verwendet: %s",
            proxy_url,
        )

        return proxy_url

    proxy_url = COMPANY_PROXY_URL or None

    if proxy_url:
        logger.debug(
            "Unternehmensproxy wird verwendet: %s",
            proxy_url,
        )
    else:
        logger.debug(
            "Keine Proxy-Verbindung. "
            "Direkte Verbindung wird verwendet."
        )

    return proxy_url


def create_http_client(
    proxy_url: str | None = None,
    ignore_ssl: bool = False,
) -> httpx.Client:
    """
    Erstellt einen synchronen HTTPX-Client.

    Args:
        proxy_url:
            Proxy-URL oder None für eine direkte Verbindung.
        ignore_ssl:
            Deaktiviert die SSL-Prüfung. Nur für Tests verwenden.
    """
    ssl_context = create_ssl_context(
        ignore_ssl=ignore_ssl
    )

    transport = httpx.HTTPTransport(
        proxy=proxy_url,
        verify=ssl_context,
        retries=RETRIES,
    )

    return httpx.Client(
        transport=transport,
        timeout=TIMEOUT,
        trust_env=False,
    )


def create_async_http_client(
    proxy_url: str | None = None,
    ignore_ssl: bool = False,
) -> httpx.AsyncClient:
    """
    Erstellt einen asynchronen HTTPX-Client.

    Args:
        proxy_url:
            Proxy-URL oder None für eine direkte Verbindung.
        ignore_ssl:
            Deaktiviert die SSL-Prüfung. Nur für Tests verwenden.
    """
    ssl_context = create_ssl_context(
        ignore_ssl=ignore_ssl
    )

    transport = httpx.AsyncHTTPTransport(
        proxy=proxy_url,
        verify=ssl_context,
        retries=RETRIES,
    )

    return httpx.AsyncClient(
        transport=transport,
        timeout=TIMEOUT,
        trust_env=False,
    )