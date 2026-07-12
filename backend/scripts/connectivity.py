"""Network diagnostics shared by standalone provider probes."""

import os
import socket
import ssl
from urllib.parse import urlparse


def report_network(url: str) -> None:
    """Report DNS, address families, TLS, proxy presence, and SSL defaults."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 443
    if host is None:
        raise ValueError("Configured URL has no host.")

    print(f"destination_host={host} port={port}")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        print(f"{name.lower()}_configured={bool(os.getenv(name))}")

    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        families = sorted({"IPv6" if item[0] == socket.AF_INET6 else "IPv4" for item in addresses})
        print(f"dns=ok address_families={','.join(families)}")
    except Exception as error:
        print(f"dns=failed exception={type(error).__module__}.{type(error).__name__} message={error}")
        return

    context = ssl.create_default_context()
    print(f"ssl_verify_mode={context.verify_mode.name} check_hostname={context.check_hostname}")
    try:
        with socket.create_connection((host, port), timeout=10) as connection:
            with context.wrap_socket(connection, server_hostname=host) as tls_socket:
                print(f"tls=ok version={tls_socket.version()}")
    except Exception as error:
        print(f"tls=failed exception={type(error).__module__}.{type(error).__name__} message={error}")
