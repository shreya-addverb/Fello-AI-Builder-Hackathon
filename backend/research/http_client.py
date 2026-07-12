"""Shared HTTPS client configuration using the operating-system trust store."""

import ssl

import httpx
import truststore


def create_async_http_client() -> httpx.AsyncClient:
    """Create a provider client that trusts certificates installed in Windows."""
    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return httpx.AsyncClient(
        verify=ssl_context,
        follow_redirects=True,
        trust_env=True,
    )
