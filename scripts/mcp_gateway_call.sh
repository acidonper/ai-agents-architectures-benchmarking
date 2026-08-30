#!/usr/bin/env bash
# Call an MCP tool on the AgentCore Gateway (/mcp) using Cognito auth.
#
# Usage:
#   source scripts/load_bedrock_agentic_env.sh
#   ./scripts/mcp_gateway_call.sh backend/Untitled-1.json
#   ./scripts/mcp_gateway_call.sh --list-tools
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# shellcheck disable=SC1091
source "$ROOT/scripts/load_bedrock_agentic_env.sh"

LIST_TOOLS=0
JSON_FILE=""
for arg in "$@"; do
  case "$arg" in
    --list-tools) LIST_TOOLS=1 ;;
    -*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *) JSON_FILE="$arg" ;;
  esac
done

PY=python3.12
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi

exec "$PY" - "$LIST_TOOLS" "$JSON_FILE" <<'PY'
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from benchmarks.bedrock_client import (
    build_request_headers,
    resolve_bedrock_gateway_url,
    resolve_gateway_urls,
)

MCP_VERSION = "2025-11-25"
list_tools = sys.argv[1] == "1"
json_file = sys.argv[2]

gw = resolve_bedrock_gateway_url()
mcp_url = resolve_gateway_urls(gw)["mcp_url"]
session_id = None

print(f"MCP URL: {mcp_url}")


def mcp_request(payload: dict) -> dict:
    global session_id
    body = json.dumps(payload).encode()
    extra = {"Content-Type": "application/json", "MCP-Protocol-Version": MCP_VERSION}
    if session_id:
        extra["Mcp-Session-Id"] = session_id
    headers = build_request_headers(
        url=mcp_url,
        method="POST",
        body=body,
        auth_mode="cognito",
        extra_headers=extra,
    )
    req = Request(mcp_url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=120) as resp:  # noqa: S310
            sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            if sid:
                session_id = sid
            raw = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", 200) or 200
    except HTTPError as exc:
        sid = exc.headers.get("Mcp-Session-Id") or exc.headers.get("mcp-session-id")
        if sid:
            session_id = sid
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    print(f"HTTP {status}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw": raw}
    print(json.dumps(data, indent=2))
    return data


# initialize
mcp_request(
    {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": MCP_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "mcp-gateway-call", "version": "1"},
        },
    }
)

# initialized notification (no id)
body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
extra = {"Content-Type": "application/json", "MCP-Protocol-Version": MCP_VERSION}
headers = build_request_headers(
    url=mcp_url,
    method="POST",
    body=body,
    auth_mode="cognito",
    extra_headers=extra,
)
try:
    urlopen(Request(mcp_url, data=body, headers=headers, method="POST"), timeout=30)  # noqa: S310
except HTTPError:
    pass

if list_tools:
    mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    sys.exit(0)

if not json_file:
    print("Usage: mcp_gateway_call.sh <json-file> | --list-tools", file=sys.stderr)
    sys.exit(2)

with open(json_file, encoding="utf-8") as fh:
    call_payload = json.load(fh)

result = mcp_request(call_payload)
if result.get("error"):
    sys.exit(1)
PY
