"""AWS Bedrock AgentCore helpers for agentic benchmarks.

Supports:
  - SigV4 auth for AgentCore Gateway (inference + MCP)
  - invoke_agent_runtime streaming via boto3
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import uuid
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MCP_PROTOCOL_VERSION = "2025-11-25"

BEDROCK_RAG_TOOL_RE = re.compile(r"retrieve|agenticretrieve", re.I)


def is_bedrock_rag_tool_name(name: str | None) -> bool:
    """True for AgentCore Gateway KB tools (Retrieve, AgenticRetrieveStream)."""
    return bool(name and BEDROCK_RAG_TOOL_RE.search(name))


def bedrock_rag_trajectory_markers(text: str) -> bool:
    """Detect Bedrock KB retrieval in trajectory / tool event strings."""
    low = (text or "").lower()
    return "file_search" in low or bool(BEDROCK_RAG_TOOL_RE.search(low))


def resolve_gateway_urls(gateway_url: str) -> dict[str, str]:
    """Derive inference and MCP URLs from a gateway base URL."""
    base = gateway_url.rstrip("/")
    if base.endswith("/mcp"):
        mcp = base
        base = base[:-4]
    elif base.endswith("/inference") or base.endswith("/inference/v1"):
        base = re.sub(r"/inference(/v1)?$", "", base)
        mcp = f"{base}/mcp"
    else:
        mcp = f"{base}/mcp"
    inference_v1 = f"{base}/inference/v1"
    return {
        "gateway_base": base,
        "inference_v1": inference_v1,
        "responses_url": f"{inference_v1}/responses",
        "models_url": f"{inference_v1}/models",
        "mcp_url": mcp,
    }


def resolve_auth_mode(explicit: str | None = None) -> str:
    mode = (explicit or os.getenv("BEDROCK_AUTH_MODE") or "iam").strip().lower()
    if mode in {"bearer", "token", "api_key", "api-key"}:
        return "bearer"
    if mode in {"cognito", "oauth", "oidc"}:
        return "cognito"
    return "iam"


def resolve_bedrock_gateway_url(explicit: str | None = None) -> str:
    """Gateway base URL from GW_HOSTNAME, BEDROCK_GATEWAY_URL, or GW_AGENTIC_ID."""
    url = (
        explicit
        or os.getenv("GW_HOSTNAME")
        or os.getenv("BEDROCK_GATEWAY_URL")
        or os.getenv("GW_AGENTIC_GATEWAY_URL")
        or ""
    ).strip().rstrip("/")
    if not url and os.getenv("GW_AGENTIC_ID"):
        region = resolve_region()
        gid = os.getenv("GW_AGENTIC_ID", "").strip()
        if gid:
            url = f"https://{gid}.gateway.bedrock-agentcore.{region}.amazonaws.com"
    return url


def resolve_region_from_gateway_url(url: str) -> str | None:
    """Extract region from *.gateway.bedrock-agentcore.{region}.amazonaws.com."""
    match = re.search(
        r"\.gateway\.bedrock-agentcore\.([a-z0-9-]+)\.amazonaws\.com",
        url,
        re.I,
    )
    return match.group(1) if match else None


def resolve_gateway_target_name() -> str | None:
    """AgentCore Gateway target name (e.g. target-quick-start-717dc2)."""
    if os.getenv("BENCH_GATEWAY_INFERENCE_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return None
    raw = (
        os.getenv("GW_AGENTIC_TARGET_QUICK_START")
        or os.getenv("BEDROCK_GATEWAY_TARGET")
        or os.getenv("GW_AGENTIC_TARGET")
        or ""
    ).strip().strip("/")
    return raw or None


def resolve_knowledge_base_id() -> str | None:
    """Bedrock KB API id (10-char) for Retrieve API."""
    return resolve_knowledge_base_api_id()


def resolve_knowledge_base_name() -> str | None:
    """Quick-start / display name (gateway MCP target prefix)."""
    raw = (
        os.getenv("GW_KNOWLEDGE_BASE_QUICK_START")
        or os.getenv("GW_KB_QUICK_START")
        or ""
    ).strip()
    return raw or None


_KB_API_ID_RE = re.compile(r"^[0-9a-zA-Z]{10}$")


def resolve_knowledge_base_api_id(region: str | None = None) -> str | None:
    """10-char knowledgeBaseId for bedrock-agent-runtime Retrieve."""
    explicit = (os.getenv("BEDROCK_KNOWLEDGE_BASE_ID") or "").strip()
    if explicit and _KB_API_ID_RE.match(explicit):
        return explicit
    name = resolve_knowledge_base_name() or explicit
    if not name:
        return None
    if _KB_API_ID_RE.match(name):
        return name
    return _lookup_knowledge_base_api_id_by_name(name, region=region)


def _lookup_knowledge_base_api_id_by_name(
    name: str,
    *,
    region: str | None = None,
) -> str | None:
    try:
        import boto3
    except ImportError:
        return None
    client = boto3.client("bedrock-agent", region_name=resolve_region(region))
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"maxResults": 25}
        if token:
            kwargs["nextToken"] = token
        resp = client.list_knowledge_bases(**kwargs)
        for summary in resp.get("knowledgeBaseSummaries") or []:
            if summary.get("name") == name:
                kb_id = summary.get("knowledgeBaseId")
                if isinstance(kb_id, str) and kb_id:
                    return kb_id
        token = resp.get("nextToken")
        if not token:
            break
    return None


def retrieve_knowledge_base(
    query: str,
    *,
    knowledge_base_id: str | None = None,
    region: str | None = None,
    number_of_results: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve passages from a managed Bedrock knowledge base (IAM / SigV4)."""
    kb_id = knowledge_base_id or resolve_knowledge_base_api_id(region=region)
    if not kb_id:
        raise RuntimeError(
            "Set BEDROCK_KNOWLEDGE_BASE_ID (10-char id) or GW_KNOWLEDGE_BASE_QUICK_START"
        )
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 required for bedrock-agent-runtime Retrieve") from exc

    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=resolve_region(region),
    )
    resp = client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {"numberOfResults": number_of_results},
        },
    )
    return list(resp.get("retrievalResults") or [])


def format_kb_retrieval_context(chunks: list[dict[str, Any]]) -> str:
    """Format Bedrock Retrieve results as prompt context (native KB RAG, not MCP)."""
    if not chunks:
        return "No relevant passages were retrieved from the Bedrock knowledge base."
    parts = ["Retrieved context from Amazon Bedrock Knowledge Base:"]
    for i, chunk in enumerate(chunks):
        text = ((chunk.get("content") or {}).get("text") or "").strip()
        if not text:
            continue
        score = chunk.get("score")
        parts.append(f"[passage {i + 1}, score={score}]\n{text}")
    return "\n\n".join(parts)


def resolve_kb_embedding_model() -> str | None:
    """Embedding model configured on the managed KB (documentation / AWS setup)."""
    raw = (
        os.getenv("BEDROCK_KB_EMBEDDING_MODEL")
        or os.getenv("GW_KB_EMBEDDING_MODEL")
        or ""
    ).strip()
    return raw or None


def resolve_kb_mcp_target_name() -> str | None:
    """Gateway connector target name for KB tools (<name>___Retrieve)."""
    raw = (
        os.getenv("BEDROCK_KB_MCP_TARGET_NAME")
        or os.getenv("GW_KB_TARGET_QUICK_START")
        or resolve_knowledge_base_name()
        or ""
    ).strip()
    return raw or None


def resolve_kb_gateway_mcp_tool_names() -> list[str]:
    """MCP tool names for managed KB Retrieve on AgentCore Gateway."""
    target = resolve_kb_mcp_target_name()
    names: list[str] = []
    if target:
        names.extend(
            [
                f"{target}___Retrieve",
                f"{target}___AgenticRetrieveStream",
            ]
        )
    names.extend(["Retrieve", "AgenticRetrieveStream"])
    # Preserve order, dedupe
    return list(dict.fromkeys(names))


def resolve_kb_retrieve_mcp_tool_name() -> str:
    """Preferred single-shot Retrieve tool name for direct MCP tools/call."""
    target = resolve_kb_mcp_target_name()
    if target:
        return f"{target}___Retrieve"
    return "Retrieve"


def resolve_gateway_target_invocation_url(
    gateway_url: str,
    target: str | None = None,
) -> str:
    """URL for POST {gateway}/{target}/invocations (AgentCore Runtime target on Gateway)."""
    name = (target or resolve_gateway_target_name() or "").strip().strip("/")
    if not name:
        raise ValueError(
            "GW_AGENTIC_TARGET_QUICK_START (or BEDROCK_GATEWAY_TARGET) is required "
            "to invoke the agent via the gateway target."
        )
    base = resolve_gateway_urls(gateway_url)["gateway_base"]
    return f"{base}/{name}/invocations"


def resolve_oauth_scope() -> str | None:
    """OAuth scope for Cognito client_credentials (never the gateway target path)."""
    explicit = (
        os.getenv("BEDROCK_OAUTH_SCOPE")
        or os.getenv("GW_AGENTIC_OAUTH_SCOPE")
        or ""
    ).strip()
    return explicit or None


def resolve_oauth_config() -> tuple[str, str, str, str | None]:
    """Cognito / OIDC client-credentials settings (AgentCore Gateway ingress)."""
    discovery = (
        os.getenv("BEDROCK_OAUTH_DISCOVERY_URL")
        or os.getenv("GW_AGENTIC_COGNITO_DISCOVERY_URL")
        or ""
    ).strip()
    client_id = (
        os.getenv("BEDROCK_OAUTH_CLIENT_ID")
        or os.getenv("GW_AGENTIC_CLIENT_ID")
        or ""
    ).strip()
    client_secret = (
        os.getenv("BEDROCK_OAUTH_CLIENT_SECRET")
        or os.getenv("GW_AGENTIC_CLIENT_SECRET")
        or ""
    ).strip()
    scope = resolve_oauth_scope()
    return discovery, client_id, client_secret, scope


_oauth_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}


def fetch_cognito_access_token(
    *,
    discovery_url: str,
    client_id: str,
    client_secret: str,
    scope: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Obtain an access token via OAuth2 client_credentials (Cognito token endpoint)."""
    now = time.time()
    cached = _oauth_token_cache.get("token")
    expires_at = float(_oauth_token_cache.get("expires_at") or 0.0)
    if isinstance(cached, str) and cached and now < expires_at - 60:
        return cached

    try:
        disc_req = Request(discovery_url, headers={"Accept": "application/json"}, method="GET")
        with urlopen(disc_req, timeout=timeout) as resp:  # noqa: S310
            discovery = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise RuntimeError(f"Failed to load OIDC discovery document: {exc}") from exc

    token_url = discovery.get("token_endpoint")
    if not token_url:
        raise RuntimeError("OIDC discovery document missing token_endpoint")

    form: dict[str, str] = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    if scope:
        form["scope"] = scope

    body = urllib.parse.urlencode(form).encode("utf-8")
    token_req = Request(
        token_url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(token_req, timeout=timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Cognito token request failed ({exc.code}): {detail[:500]}"
        ) from exc
    except (URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise RuntimeError(f"Cognito token request failed: {exc}") from exc

    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError(f"Cognito token response missing access_token: {payload}")

    expires_in = payload.get("expires_in")
    try:
        ttl = float(expires_in) if expires_in is not None else 3600.0
    except (TypeError, ValueError):
        ttl = 3600.0
    _oauth_token_cache["token"] = token
    _oauth_token_cache["expires_at"] = now + ttl
    return token


def resolve_cognito_bearer_token() -> str:
    discovery, client_id, client_secret, scope = resolve_oauth_config()
    missing = [
        name
        for name, val in (
            ("BEDROCK_OAUTH_DISCOVERY_URL / GW_AGENTIC_COGNITO_DISCOVERY_URL", discovery),
            ("BEDROCK_OAUTH_CLIENT_ID / GW_AGENTIC_CLIENT_ID", client_id),
            ("BEDROCK_OAUTH_CLIENT_SECRET / GW_AGENTIC_CLIENT_SECRET", client_secret),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Cognito auth requires: " + ", ".join(missing)
        )
    return fetch_cognito_access_token(
        discovery_url=discovery,
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
    )


def resolve_bearer_token(api_key: str | None = None) -> str | None:
    token = (
        api_key
        or os.getenv("BEDROCK_GATEWAY_TOKEN")
        or os.getenv("BENCH_API_KEY")
        or os.getenv("BEDROCK_API_KEY")
        or ""
    ).strip()
    return token or None


def resolve_region(explicit: str | None = None) -> str:
    return (
        explicit
        or os.getenv("BEDROCK_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def sign_headers(
    *,
    url: str,
    method: str,
    body: bytes | None,
    region: str,
    service: str = "bedrock-agentcore",
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return SigV4-signed headers for an AgentCore Gateway HTTP request."""
    try:
        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for IAM auth to Bedrock AgentCore Gateway. "
            "Install boto3 or set BEDROCK_AUTH_MODE=bearer with a gateway token."
        ) from exc

    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        raise RuntimeError(
            "No AWS credentials found for SigV4 signing. "
            "Configure AWS_PROFILE / AWS_ACCESS_KEY_ID or use bearer auth."
        )
    headers = {"Accept": "application/json", **(extra_headers or {})}
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    request = AWSRequest(method=method, url=url, data=body, headers=headers)
    frozen = creds.get_frozen_credentials()
    SigV4Auth(frozen, service, region).add_auth(request)
    return dict(request.headers)


def build_request_headers(
    *,
    url: str,
    method: str,
    body: bytes | None,
    region: str | None = None,
    auth_mode: str | None = None,
    bearer_token: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build Authorization headers for gateway inference or MCP calls."""
    mode = resolve_auth_mode(auth_mode)
    headers = dict(extra_headers or {})
    if mode == "cognito":
        token = resolve_cognito_bearer_token()
        headers["Authorization"] = f"Bearer {token}"
        return headers
    if mode == "bearer":
        token = resolve_bearer_token(bearer_token)
        if not token:
            raise RuntimeError(
                "Bearer / api-key auth selected but no token found. "
                "Set BEDROCK_API_KEY, BEDROCK_GATEWAY_TOKEN, or BENCH_API_KEY."
            )
        headers["Authorization"] = f"Bearer {token}"
        return headers
    signed = sign_headers(
        url=url,
        method=method,
        body=body,
        region=resolve_region(region),
        extra_headers=headers,
    )
    return signed


def iter_runtime_sse_lines(raw_line: bytes) -> Iterator[str]:
    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
    if not line:
        return
    if line.startswith("data: "):
        yield line[6:]
    elif line.startswith("data:"):
        yield line[5:].lstrip()
    else:
        yield line


def extract_runtime_text(chunk: str) -> str | None:
    """Best-effort text extraction from AgentCore runtime stream chunks."""
    chunk = (chunk or "").strip()
    if not chunk or chunk == "[DONE]":
        return None
    try:
        data = json.loads(chunk)
    except json.JSONDecodeError:
        return chunk if chunk and not chunk.startswith("{") else None

    if isinstance(data, str):
        return data or None
    if not isinstance(data, dict):
        return None

    for key in ("text", "output_text", "content", "message", "delta"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict):
            nested = val.get("text") or val.get("content")
            if isinstance(nested, str) and nested:
                return nested

    output = data.get("output")
    if isinstance(output, str) and output:
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                for part in item.get("content") or []:
                    if isinstance(part, dict):
                        t = part.get("text") or part.get("content")
                        if isinstance(t, str):
                            parts.append(t)
        if parts:
            return "".join(parts)

    event = data.get("event") or data.get("type")
    if isinstance(event, str) and "delta" in event.lower():
        delta = data.get("delta") or data.get("data")
        if isinstance(delta, str):
            return delta
        if isinstance(delta, dict):
            t = delta.get("text") or delta.get("content")
            if isinstance(t, str):
                return t
    return None


def invoke_agent_runtime_stream(
    *,
    agent_runtime_arn: str,
    payload: dict[str, Any],
    runtime_session_id: str,
    qualifier: str | None = None,
    region: str | None = None,
) -> tuple[int, str, Iterator[bytes]]:
    """Invoke AgentCore Runtime and return (status_hint, content_type, line iterator)."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for bedrock-runtime target.") from exc

    client = boto3.client("bedrock-agentcore", region_name=resolve_region(region))
    body = json.dumps(payload).encode("utf-8")
    kwargs: dict[str, Any] = {
        "agentRuntimeArn": agent_runtime_arn,
        "runtimeSessionId": runtime_session_id,
        "payload": body,
    }
    if qualifier:
        kwargs["qualifier"] = qualifier

    response = client.invoke_agent_runtime(**kwargs)
    content_type = str(response.get("contentType") or "")
    stream = response.get("response")
    if stream is None:
        raise RuntimeError("invoke_agent_runtime returned no response body")

    def _iter() -> Iterator[bytes]:
        for line in stream.iter_lines():
            if line:
                yield line

    status_hint = 200
    return status_hint, content_type, _iter()


def resolve_agent_runtime_arn() -> str | None:
    raw = (
        os.getenv("BEDROCK_AGENT_RUNTIME_ARN")
        or os.getenv("GW_AGENTIC_RUNTIME_ARN")
        or ""
    ).strip()
    return raw or None


def resolve_runtime_invocations_url(
    arn: str | None = None,
    *,
    qualifier: str | None = None,
    region: str | None = None,
) -> str:
    """HTTPS InvokeAgentRuntime URL for MCP JSON-RPC (Cognito Bearer)."""
    runtime_arn = (arn or resolve_agent_runtime_arn() or "").strip()
    if not runtime_arn:
        raise ValueError("Set BEDROCK_AGENT_RUNTIME_ARN for runtime MCP")
    q = (qualifier or os.getenv("BEDROCK_RUNTIME_QUALIFIER") or "DEFAULT").strip() or "DEFAULT"
    encoded = urllib.parse.quote(runtime_arn, safe="")
    return (
        f"https://bedrock-agentcore.{resolve_region(region)}.amazonaws.com"
        f"/runtimes/{encoded}/invocations?qualifier={urllib.parse.quote(q, safe='')}"
    )


def parse_mcp_response_body(raw: str) -> dict[str, Any]:
    """Parse JSON-RPC from SSE (`event: message` / `data:`) or a JSON body."""
    text = (raw or "").strip()
    if not text:
        return {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            payload = stripped[5:].strip()
            if payload and payload != "[DONE]":
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict):
                    return data
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text[:2000]}
    return data if isinstance(data, dict) else {"raw": data}


def mcp_tool_text(result: dict[str, Any]) -> str:
    """Flatten MCP tools/call result content to a string for the LLM."""
    if result.get("error"):
        return json.dumps(result["error"])
    inner = result.get("result") if isinstance(result.get("result"), dict) else result
    if not isinstance(inner, dict):
        return json.dumps(result)[:8000]
    parts: list[str] = []
    for item in inner.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            t = item.get("text")
            if isinstance(t, str):
                parts.append(t)
    if parts:
        return "\n".join(parts)
    structured = inner.get("structuredContent")
    if structured is not None:
        return json.dumps(structured) if not isinstance(structured, str) else structured
    return json.dumps(inner)[:8000]


class AgentCoreRuntimeMcpClient:
    """MCP JSON-RPC client for AgentCore Runtime InvokeAgentRuntime."""

    def __init__(
        self,
        url: str,
        *,
        auth_mode: str = "cognito",
        timeout: float = 120.0,
    ) -> None:
        self.url = url
        self.auth_mode = auth_mode
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        self._rpc_id = 0

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        notify: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notify:
            self._rpc_id += 1
            payload["id"] = self._rpc_id
        if params is not None:
            payload["params"] = params
        body = json.dumps(payload).encode("utf-8")
        extra = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "MCP-Session-Id": self.session_id,
            "Mcp-Method": method,
        }
        if method == "tools/call" and isinstance(params, dict) and params.get("name"):
            extra["Mcp-Name"] = str(params["name"])
        headers = build_request_headers(
            url=self.url,
            method="POST",
            body=body,
            auth_mode=self.auth_mode,
            extra_headers=extra,
        )
        req = Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
                if sid:
                    self.session_id = sid
                raw = resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            sid = exc.headers.get("Mcp-Session-Id") or exc.headers.get("mcp-session-id")
            if sid:
                self.session_id = sid
            raw = exc.read().decode("utf-8", errors="replace")
            data = parse_mcp_response_body(raw)
            if "error" not in data:
                data["error"] = {"code": exc.code, "message": raw[:500]}
            return data
        return parse_mcp_response_body(raw)

    def initialize(self) -> dict[str, Any]:
        result = self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agentic-bench", "version": "1"},
            },
        )
        try:
            self.request("notifications/initialized", {}, notify=True)
        except Exception:  # noqa: BLE001
            pass
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        data = self.request("tools/list", {})
        tools = (data.get("result") or {}).get("tools") or []
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments or {}})


def mcp_tools_to_converse_specs(
    tools: list[dict[str, Any]],
    allowed: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert MCP tools/list entries to Bedrock Converse toolSpec list."""
    allowed_set = set(allowed) if allowed else None
    specs: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        if allowed_set is not None and name not in allowed_set:
            continue
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        description = tool.get("description") or tool.get("title") or name
        if not isinstance(description, str):
            description = name
        specs.append(
            {
                "toolSpec": {
                    "name": name,
                    "description": description[:1024],
                    "inputSchema": {"json": schema},
                }
            }
        )
    return specs


def converse_with_tools(
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    region: str | None = None,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """One Bedrock Converse turn (IAM). Model id from BENCH_MODEL_INFERENCE_PROFILE_ID."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for Bedrock Converse inference") from exc
    client = boto3.client("bedrock-runtime", region_name=resolve_region(region))
    kwargs: dict[str, Any] = {
        "modelId": model_id,
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.2},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    if tools:
        kwargs["toolConfig"] = {"tools": tools, "toolChoice": {"auto": {}}}
    return client.converse(**kwargs)
