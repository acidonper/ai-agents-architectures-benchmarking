"""Smoke test: Cognito token + AgentCore Gateway ingress (MCP) auth.

Does not invoke the agent runtime target (that requires aligned inbound-auth in AWS).
"""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from benchmarks.bedrock_client import (
    build_request_headers,
    resolve_bedrock_gateway_url,
    resolve_gateway_target_invocation_url,
    resolve_gateway_target_name,
    resolve_gateway_urls,
)


def _probe(url: str, method: str = "POST", body: bytes = b"{}", extra: dict | None = None) -> tuple[int, str]:
    headers = build_request_headers(
        url=url,
        method=method,
        body=body,
        auth_mode="cognito",
        extra_headers={"Content-Type": "application/json", **(extra or {})},
    )
    req = Request(url, data=body if method != "GET" else None, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            raw = resp.read(500).decode("utf-8", errors="replace")
            return getattr(resp, "status", 200) or 200, raw
    except HTTPError as exc:
        raw = exc.read(500).decode("utf-8", errors="replace")
        return exc.code, raw


def main() -> int:
    gateway = resolve_bedrock_gateway_url()
    if not gateway:
        print("ERROR: Set GW_HOSTNAME or BEDROCK_GATEWAY_URL", file=sys.stderr)
        return 2

    urls = resolve_gateway_urls(gateway)
    mcp_url = urls["mcp_url"]
    target = resolve_gateway_target_name()

    print(f"Gateway: {gateway}")
    print(f"MCP:     {mcp_url}")
    if target:
        invoke = resolve_gateway_target_invocation_url(gateway, target)
        print(f"Target:  {target} → {invoke}")

    # 1) Cognito client_credentials (build_request_headers fetches token)
    try:
        build_request_headers(url=mcp_url, method="POST", body=b"{}", auth_mode="cognito")
        print("OK  Cognito token (client_credentials)")
    except Exception as exc:
        print(f"FAIL Cognito token: {exc}", file=sys.stderr)
        return 1

    # 2) Gateway MCP ingress accepts Bearer
    mcp_body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "1"},
            },
        }
    ).encode()
    status, raw = _probe(
        mcp_url,
        body=mcp_body,
        extra={"MCP-Protocol-Version": "2025-11-25"},
    )
    if status < 400 or "protocol version" in raw.lower():
        print(f"OK  Gateway MCP ingress auth (HTTP {status})")
    else:
        print(f"FAIL Gateway MCP: HTTP {status} {raw}", file=sys.stderr)
        return 1

    # 3) Optional: runtime target (often fails until inbound-auth discoveryUrl matches iss)
    if target and os.getenv("SMOKE_PROBE_AGENT_TARGET", "").strip().lower() in {"1", "true", "yes"}:
        invoke = resolve_gateway_target_invocation_url(gateway, target)
        body = json.dumps({"prompt": "ping"}).encode()
        status, raw = _probe(invoke, body=body)
        if status < 400:
            print(f"OK  Runtime target invocation (HTTP {status})")
        elif "iss" in raw.lower() and "mismatch" in raw.lower():
            print(
                "WARN Runtime target: iss mismatch — align AgentCore runtime inbound-auth "
                "discoveryUrl with GW_AGENTIC_COGNITO_DISCOVERY_URL in AWS console",
                file=sys.stderr,
            )
            return 1
        else:
            print(f"FAIL Runtime target: HTTP {status} {raw}", file=sys.stderr)
            return 1

    if target:
        print(
            "Note: agent runtime target not probed. Use SMOKE_PROBE_AGENT_TARGET=1 or "
            "./test.sh --agent-target after fixing runtime inbound auth."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
