import logging
import socket
import ssl

import httpx

from legal_system_rag.config import (
    CERT_FILE,
    COMPANY_PROXY_URL,
    IGNORE_SSL,
    PX_HOST,
    PX_PORT,
    RETRIES,
    TIMEOUT,
)

logger = logging.getLogger(__name__)


def check_if_px_is_running() -> bool:
    """
    Check whether the local PX proxy is reachable.
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
    ignore_ssl: bool = IGNORE_SSL,
) -> ssl.SSLContext:
    """
    Create the SSL context using the configured CA bundle.

    Args:
        ignore_ssl:
            Disable certificate and hostname verification.
            Only use for tests.
    """
    if not CERT_FILE.exists():
        raise FileNotFoundError(
            f"CA certificate was not found: {CERT_FILE}"
        )

    context = ssl.create_default_context(
        cafile=str(CERT_FILE)
    )

    if ignore_ssl:
        logger.warning(
            "SSL certificate verification is disabled. "
            "Only use this for tests!"
        )

        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    return context


def get_proxy_url() -> str | None:
    """
    Determine which proxy to use.

    Priority:
        1. Local PX proxy.
        2. Company proxy.
        3. Direct connection.
    """
    if check_if_px_is_running():
        proxy_url = f"http://{PX_HOST}:{PX_PORT}"

        logger.debug(
            "Using local PX proxy: %s",
            proxy_url,
        )

        return proxy_url

    proxy_url = COMPANY_PROXY_URL or None

    if proxy_url:
        logger.debug(
            "Using company proxy: %s",
            proxy_url,
        )
    else:
        logger.debug(
            "No proxy connection. "
            "Using direct connection."
        )

    return proxy_url


def create_http_client(
    proxy_url: str | None = None,
    ignore_ssl: bool = IGNORE_SSL,
) -> httpx.Client:
    """
    Create a synchronous HTTPX client.

    Args:
        proxy_url:
            Proxy URL or None for a direct connection.
        ignore_ssl:
            Disable SSL verification. Only use for tests.
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
    ignore_ssl: bool = IGNORE_SSL,
) -> httpx.AsyncClient:
    """
    Create an asynchronous HTTPX client.

    Args:
        proxy_url:
            Proxy URL or None for a direct connection.
        ignore_ssl:
            Disable SSL verification. Only use for tests.
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