"""HTTP MCP client for any MCP endpoint (API Gateway or AgentCore Gateway)."""

from __future__ import annotations

import argparse
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MCP_VERSION = "2025-11-25"


def mcp_post(
    url: str,
    payload: dict,
    *,
    bearer: str | None = None,
    session_id: str | None = None,
    timeout: float = 120.0,
) -> tuple[int, dict, str | None]:
    body = json.dumps(payload).encode()
    headers = {
        "Content-Type": "application/json",
        "MCP-Protocol-Version": MCP_VERSION,
    }
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            raw = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200) or 200
    except HTTPError as exc:
        sid = exc.headers.get("Mcp-Session-Id") or exc.headers.get("mcp-session-id")
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code

    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {"raw": raw}

    print(f"HTTP {status}")
    print(json.dumps(data, indent=2)[:12000])
    return status, data, sid


def main() -> int:
    parser = argparse.ArgumentParser(description="MCP HTTP call (initialize + optional JSON payload)")
    parser.add_argument("url", help="MCP endpoint URL")
    parser.add_argument("json_file", nargs="?", help="JSON-RPC payload (e.g. tools/call)")
    parser.add_argument("--list-tools", action="store_true")
    parser.add_argument("--cognito", action="store_true", help="Use Cognito bearer from env")
    parser.add_argument("--bearer", default="", help="Explicit Bearer token")
    args = parser.parse_args()

    bearer = args.bearer.strip() or None
    if args.cognito:
        from benchmarks.bedrock_client import resolve_cognito_bearer_token

        bearer = resolve_cognito_bearer_token()

    session: str | None = None
    print(f"MCP URL: {args.url}")
    if bearer:
        print(f"Auth: Bearer ({len(bearer)} chars)")

    _, _, session = mcp_post(
        args.url,
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mcp-http-call", "version": "1"},
            },
        },
        bearer=bearer,
    )

    # notifications/initialized
    body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    headers = {"Content-Type": "application/json", "MCP-Protocol-Version": MCP_VERSION}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if session:
        headers["Mcp-Session-Id"] = session
    try:
        urlopen(Request(args.url, data=body, headers=headers, method="POST"), timeout=30)  # noqa: S310
    except HTTPError:
        pass

    if args.list_tools:
        status, data, _ = mcp_post(
            args.url,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            bearer=bearer,
            session_id=session,
        )
        return 0 if status < 400 and not data.get("error") else 1

    if not args.json_file:
        return 0

    with open(args.json_file, encoding="utf-8") as fh:
        payload = json.load(fh)

    status, data, _ = mcp_post(args.url, payload, bearer=bearer, session_id=session)
    if status >= 400 or data.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
