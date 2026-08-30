#!/usr/bin/env python3
"""Agentic MCP benchmark: LLM + MCP latency, Phoenix semantic/trajectory evals.

Targets:
  llama-stack       → POST {LLAMA_STACK}/v1/responses with tools=[{type:mcp,...}]
  stackchat         → POST {STACKCHAT}/api/chat with enable_mcp=true
  bedrock-gateway   → POST {GATEWAY}/{GW_AGENTIC_TARGET}/invocations + Cognito Bearer
                      (or /inference/v1/responses if no target is set)
  bedrock-runtime   → LLM = Bedrock Converse (BENCH_MODEL_INFERENCE_PROFILE_ID);
                      MCP = AgentCore Runtime InvokeAgentRuntime JSON-RPC;
                      RAG = bedrock-agent-runtime Retrieve (BENCH_RAG_MODE=bedrock_kb)

Scenarios (--scenario):
  list    — list issues in SUP (default prompt / instructions)
  create  — create a Jira issue from system + user prompt files
  rag     — RAG (file_search) + MCP together; measures rag_time_s and mcp_time_s

With --phoenix:
  - orchestration_time_s (E2E agent turn)
  - semantic success (LLM-as-judge via Phoenix evals)
  - trajectory correctness (LLM-as-judge)
  - combined success_rate
  - optional local Phoenix UI / OTEL export
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bedrock_client import (
    AgentCoreRuntimeMcpClient,
    converse_with_tools,
    extract_runtime_text,
    format_kb_retrieval_context,
    is_bedrock_rag_tool_name,
    mcp_tool_text,
    mcp_tools_to_converse_specs,
    resolve_agent_runtime_arn,
    resolve_auth_mode,
    resolve_bedrock_gateway_url,
    resolve_gateway_target_invocation_url,
    resolve_gateway_target_name,
    retrieve_knowledge_base,
    resolve_kb_embedding_model,
    resolve_kb_gateway_mcp_tool_names,
    resolve_knowledge_base_id,
    resolve_knowledge_base_api_id,
    resolve_knowledge_base_name,
    resolve_gateway_urls,
    resolve_runtime_invocations_url,
    build_request_headers,
)

DEFAULT_PROMPT = "List Jira issues in project SUP. Summarize key, status, and summary."
DEFAULT_LIST_INSTRUCTIONS = (
    "You are a Jira assistant. Always use MCP tools. For project listings, "
    "call jira_search (JQL like project = SUP) or jira_get_project_issues. "
    "Never invent issue keys. Summarize only issues returned by tools "
    "(key, status, summary)."
)
DEFAULT_CREATE_SYSTEM = Path("benchmarks/prompts/jira_create/system.txt")
DEFAULT_CREATE_USER = Path("benchmarks/prompts/jira_create/user.txt")
DEFAULT_RAG_SYSTEM = Path("benchmarks/prompts/rag/system.txt")
DEFAULT_RAG_USER = Path("benchmarks/prompts/rag/user.txt")
DEFAULT_RAG_PROMPT = (
    "Answer using the knowledge base. Cite specific facts from retrieved documents."
)
PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")


def load_prompt_file(path: Path, *, require_filled: bool = True) -> str:
    """Load a prompt file; strip # comment lines; optionally reject <PLACEHOLDER>s."""
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError(f"Prompt file is empty after stripping comments: {path}")
    if require_filled:
        leftover = sorted(set(PLACEHOLDER_RE.findall(text)))
        if leftover:
            raise ValueError(
                f"Unfilled placeholders in {path}: {', '.join(leftover)}. "
                "Edit the file before running the create-issue bench "
                "(see benchmarks/prompts/jira_create/README.md)."
            )
    return text


@dataclass
class TrialResult:
    ok: bool
    latency_s: float
    status_code: int | None = None
    response_id: str | None = None
    output_text: str = ""
    tool_events: list[str] = field(default_factory=list)
    mcp_list_tools: int = 0
    mcp_calls: int = 0
    error: str | None = None
    raw_output: Any = None
    trajectory: str = ""
    trial_id: int | None = None
    mcp_tool_responses: list[dict[str, Any]] = field(default_factory=list)
    jira_issues: list[dict[str, Any]] = field(default_factory=list)
    # Streaming / token metrics
    streamed: bool = False
    ttft_s: float | None = None
    itl_s: float | None = None
    itl_s_median: float | None = None
    itl_s_p95: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    output_tokens_estimated: int | None = None
    stream_delta_count: int = 0
    tokens_per_second: float | None = None
    # RAG / file_search metrics
    rag_calls: int = 0
    rag_time_s: float | None = None
    time_to_rag_s: float | None = None
    rag_search_spans_s: list[float] = field(default_factory=list)
    # MCP metrics
    mcp_time_s: float | None = None
    time_to_mcp_s: float | None = None
    mcp_call_spans_s: list[float] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        """End-to-end time for this request (LLM + tools orchestration)."""
        return self.latency_s

    @property
    def tokens_reported(self) -> int | None:
        if self.output_tokens is not None:
            return self.output_tokens
        return self.output_tokens_estimated


def _load_dotenv(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        val = value.strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = val


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 180.0,
) -> tuple[int, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req_headers = {"Accept": "application/json", **(headers or {})}
    if body is not None:
        req_headers["Content-Type"] = "application/json"
    req = Request(url, data=body, headers=req_headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            data = json.loads(raw) if raw else None
            return resp.status, data
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw else {"detail": str(exc)}
        except json.JSONDecodeError:
            data = {"detail": raw or str(exc)}
        return exc.code, data
    except URLError as exc:
        raise ConnectionError(str(exc.reason or exc)) from exc


def _estimate_tokens(text: str) -> int:
    """Rough token estimate when the API does not return usage."""
    text = (text or "").strip()
    if not text:
        return 0
    # Prefer whitespace tokens; fall back to ~4 chars/token.
    words = len(text.split())
    chars = max(1, (len(text) + 3) // 4)
    return max(words, chars)


def _extract_usage(payload: Any) -> dict[str, int | None]:
    """Pull token counts from Responses / Chat Completions style usage blobs."""
    empty = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    if not isinstance(payload, dict):
        return empty
    usage = payload.get("usage")
    if not isinstance(usage, dict) and isinstance(payload.get("response"), dict):
        usage = payload["response"].get("usage")
    if not isinstance(usage, dict):
        return empty

    def _pick(*keys: str) -> int | None:
        for key in keys:
            val = usage.get(key)
            if isinstance(val, (int, float)) and val >= 0:
                return int(val)
        return None

    inp = _pick("input_tokens", "prompt_tokens", "inputTokens")
    out = _pick("output_tokens", "completion_tokens", "outputTokens")
    total = _pick("total_tokens", "totalTokens")
    if total is None and inp is not None and out is not None:
        total = inp + out
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": total}


def _metric_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "p95": None, "sum": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sum": sum(values),
        "p95": (
            statistics.quantiles(values, n=20)[18]
            if len(values) >= 2
            else values[0]
        ),
    }


def _compute_itl(
    delta_times: list[float],
    *,
    output_tokens: int | None,
    tool_gap_s: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    """Inter-token latency from streamed text delta arrival times.

    Long gaps (typical MCP tool waits) are excluded from pairwise ITL.
    When output_tokens is known and generation span has ≥2 tokens, also
    derive a token-normalized ITL from (t_last - t_first) / (n - 1) using
    only the first contiguous generation segment after TTFT.
    """
    if len(delta_times) < 2:
        return None, None, None

    pairwise: list[float] = []
    for prev, cur in zip(delta_times, delta_times[1:]):
        gap = cur - prev
        if 0 < gap < tool_gap_s:
            pairwise.append(gap)

    # Token-normalized ITL over the first contiguous generation burst.
    token_itl: float | None = None
    if output_tokens and output_tokens > 1:
        segment = [delta_times[0]]
        for t in delta_times[1:]:
            if t - segment[-1] >= tool_gap_s:
                break
            segment.append(t)
        span = segment[-1] - segment[0]
        if span > 0:
            token_itl = span / (output_tokens - 1)

    samples = pairwise
    if token_itl is not None:
        # Prefer token-normalized mean when available; keep pairwise for median/p95.
        mean = token_itl
    elif samples:
        mean = statistics.mean(samples)
    else:
        mean = None

    if not samples and token_itl is None:
        return None, None, None
    median = statistics.median(samples) if samples else token_itl
    p95 = (
        statistics.quantiles(samples, n=20)[18]
        if len(samples) >= 2
        else (samples[0] if samples else token_itl)
    )
    return mean, median, p95


def _attach_token_metrics(
    result: TrialResult,
    *,
    started: float,
    delta_times: list[float],
    delta_texts: list[str],
    usage: dict[str, int | None] | None,
    streamed: bool,
) -> TrialResult:
    usage = usage or {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    text = result.output_text or "".join(delta_texts)
    estimated = _estimate_tokens(text) if text else 0
    out_tokens = usage.get("output_tokens")
    if out_tokens is None and estimated:
        result.output_tokens_estimated = estimated
    else:
        result.output_tokens_estimated = estimated if out_tokens is None else None

    result.streamed = streamed
    result.stream_delta_count = len(delta_times)
    result.input_tokens = usage.get("input_tokens")
    result.output_tokens = out_tokens
    result.total_tokens = usage.get("total_tokens")
    if result.total_tokens is None and result.input_tokens is not None and out_tokens is not None:
        result.total_tokens = result.input_tokens + out_tokens

    reported = out_tokens if out_tokens is not None else (estimated or None)

    if delta_times:
        result.ttft_s = delta_times[0] - started
        mean, median, p95 = _compute_itl(
            delta_times,
            output_tokens=reported,
        )
        result.itl_s = mean
        result.itl_s_median = median
        result.itl_s_p95 = p95
        gen_span = delta_times[-1] - delta_times[0]
        if reported and reported > 0 and gen_span > 0:
            result.tokens_per_second = reported / gen_span
        elif reported and reported > 0 and result.latency_s > 0:
            # Fallback: e2e (includes MCP) — still useful when only one delta.
            result.tokens_per_second = reported / result.latency_s
    return result


def _iter_sse_events(
    resp: Any,
) -> Any:
    """Yield (event_name, data_dict_or_str) from an SSE HTTP response body."""
    event_name = "message"
    data_lines: list[str] = []
    while True:
        raw = resp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                data_str = "\n".join(data_lines)
                data_lines = []
                name = event_name
                event_name = "message"
                if data_str.strip() == "[DONE]":
                    yield name, "[DONE]"
                    continue
                try:
                    yield name, json.loads(data_str)
                except json.JSONDecodeError:
                    yield name, data_str
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


def _delta_text_from_event(event_name: str, payload: Any) -> str | None:
    """Extract streamed assistant text from Responses API or StackChat SSE events."""
    if not isinstance(payload, dict):
        return None
    typ = str(payload.get("type") or event_name or "")
    if event_name == "delta" or typ in {
        "response.output_text.delta",
        "response.text.delta",
        "content.delta",
        "response.content_part.delta",
    }:
        delta = (
            payload.get("delta")
            or payload.get("text")
            or payload.get("content")
            or ""
        )
        if isinstance(delta, dict):
            delta = delta.get("text") or delta.get("content") or delta.get("delta") or ""
        text = str(delta)
        return text if text else None
    return None


def _is_tool_stream_event(event_name: str, payload: Any) -> bool:
    typ = ""
    if isinstance(payload, dict):
        typ = str(payload.get("type") or "")
    blob = f"{event_name} {typ}".lower()
    if "delta" in blob or "arguments" in blob:
        return False
    return any(
        x in blob
        for x in (
            "mcp",
            "tool",
            "function_call",
            "file_search",
        )
    )


def _stream_request(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None,
    timeout: float,
    stackchat: bool = False,
) -> TrialResult:
    """POST with stream=true and collect TTFT / ITL / token metrics from SSE."""
    body = json.dumps(payload).encode("utf-8")
    req_headers = {
        "Accept": "text/event-stream, application/json",
        "Content-Type": "application/json",
        **(headers or {}),
    }
    req = Request(url, data=body, headers=req_headers, method="POST")
    started = time.perf_counter()
    delta_times: list[float] = []
    delta_texts: list[str] = []
    events: list[str] = []
    final_payload: dict[str, Any] | None = None
    usage: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    status = 0
    accumulated = ""
    rag_open_starts: list[float] = []
    rag_spans: list[float] = []
    time_to_rag: float | None = None
    mcp_open_starts: list[float] = []
    mcp_spans: list[float] = []
    time_to_mcp: float | None = None

    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            status = getattr(resp, "status", 200) or 200
            content_type = (resp.headers.get("Content-Type") or "").lower()
            # Some stacks ignore stream=true and return a single JSON body.
            if "application/json" in content_type and "event-stream" not in content_type:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw) if raw else None
                latency = time.perf_counter() - started
                result = _finish_trial(
                    status=status,
                    data=data,
                    latency=latency,
                    stackchat=stackchat,
                )
                return _attach_token_metrics(
                    result,
                    started=started,
                    delta_times=[],
                    delta_texts=[],
                    usage=_extract_usage(data),
                    streamed=False,
                )

            for event_name, data in _iter_sse_events(resp):
                now = time.perf_counter()
                if data == "[DONE]":
                    break
                if isinstance(data, dict):
                    u = _extract_usage(data)
                    if u["output_tokens"] is not None or u["input_tokens"] is not None:
                        usage = {k: u[k] if u[k] is not None else usage[k] for k in usage}
                    if isinstance(data.get("response"), dict):
                        final_payload = data["response"]
                        u2 = _extract_usage(final_payload)
                        usage = {
                            k: u2[k] if u2[k] is not None else usage[k] for k in usage
                        }
                    typ = str(data.get("type") or event_name)
                    if typ in {"response.completed", "response.done"} or event_name in {
                        "message",
                        "done",
                    }:
                        if event_name == "message" and stackchat:
                            final_payload = data
                        elif typ in {"response.completed", "response.done"}:
                            final_payload = data.get("response") or data

                    # RAG timing from file_search_call or Bedrock KB MCP tools (Retrieve).
                    if _is_rag_stream_event(event_name, data):
                        low = typ.lower()
                        rag_label = (
                            "retrieve_call"
                            if is_bedrock_rag_tool_name(_mcp_tool_name_from_event(data))
                            else "file_search_call"
                        )
                        events.append(rag_label)
                        if final_payload is None:
                            final_payload = {"output": []}
                        raw_events = final_payload.setdefault("_stream_events", [])
                        if isinstance(raw_events, list):
                            raw_events.append(data)
                        if ".completed" in low or low.endswith("file_search_call.completed"):
                            if rag_open_starts:
                                span_start = rag_open_starts.pop(0)
                                rag_spans.append(max(0.0, now - span_start))
                            elif time_to_rag is not None:
                                rag_spans.append(max(0.0, now - (started + time_to_rag)))
                        elif "searching" in low:
                            if time_to_rag is None:
                                time_to_rag = now - started
                            if rag_open_starts:
                                rag_open_starts[-1] = now
                            else:
                                rag_open_starts.append(now)
                        elif "in_progress" in low:
                            if time_to_rag is None:
                                time_to_rag = now - started
                            if not rag_open_starts:
                                rag_open_starts.append(now)
                        continue

                    # MCP call timing (ignore arguments.done — only in_progress→completed).
                    if _is_mcp_call_stream_event(event_name, data):
                        if is_bedrock_rag_tool_name(_mcp_tool_name_from_event(data)):
                            continue
                        low = typ.lower()
                        events.append("mcp_call")
                        if final_payload is None:
                            final_payload = {"output": []}
                        raw_events = final_payload.setdefault("_stream_events", [])
                        if isinstance(raw_events, list):
                            raw_events.append(data)
                        if "arguments" in low:
                            continue
                        if ".completed" in low or low.endswith("mcp_call.completed"):
                            if mcp_open_starts:
                                span_start = mcp_open_starts.pop(0)
                                mcp_spans.append(max(0.0, now - span_start))
                            elif time_to_mcp is not None:
                                mcp_spans.append(max(0.0, now - (started + time_to_mcp)))
                        elif "in_progress" in low:
                            if time_to_mcp is None:
                                time_to_mcp = now - started
                            if not mcp_open_starts:
                                mcp_open_starts.append(now)
                        continue

                    delta = _delta_text_from_event(event_name, data)
                    if delta is not None:
                        delta_times.append(now)
                        delta_texts.append(delta)
                        accumulated += delta
                    elif _is_tool_stream_event(event_name, data):
                        typ = str(data.get("type") or event_name)
                        norm = _normalize_tool_event(typ) or typ
                        events.append(norm)
                        # Keep collecting tool structure for trajectory.
                        if final_payload is None:
                            final_payload = {"output": []}
                        # Stash raw tool event under a list for trajectory helpers.
                        raw_events = final_payload.setdefault("_stream_events", [])
                        if isinstance(raw_events, list):
                            raw_events.append(data)
                    else:
                        # Capture any mcp_* typed objects nested in stream frames.
                        nested = _collect_tool_events(data)
                        events.extend(nested)
                elif event_name == "error":
                    latency = time.perf_counter() - started
                    return TrialResult(
                        ok=False,
                        latency_s=latency,
                        status_code=status,
                        error=str(data)[:1000],
                        streamed=True,
                    )
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw else {"detail": str(exc)}
        except json.JSONDecodeError:
            data = {"detail": raw or str(exc)}
        latency = time.perf_counter() - started
        # Non-stream error body (e.g. 4xx before SSE starts).
        if isinstance(data, dict) and not str(exc.headers.get("Content-Type", "")).startswith(
            "text/event-stream"
        ):
            result = _finish_trial(
                status=exc.code,
                data=data,
                latency=latency,
                stackchat=stackchat,
            )
            return _attach_token_metrics(
                result,
                started=started,
                delta_times=[],
                delta_texts=[],
                usage=_extract_usage(data),
                streamed=False,
            )
        return TrialResult(
            ok=False,
            latency_s=latency,
            status_code=exc.code,
            error=str(data)[:1000],
            streamed=True,
        )
    except URLError as exc:
        return TrialResult(
            ok=False,
            latency_s=time.perf_counter() - started,
            error=str(exc.reason or exc),
            streamed=True,
        )

    latency = time.perf_counter() - started
    # Close any open tool spans that never saw a completed event (zero-duration
    # calls often emit in_progress+completed in the same tick, or only in_progress).
    end_mark = delta_times[0] if delta_times else (started + latency)
    while rag_open_starts:
        rag_spans.append(max(0.0, end_mark - rag_open_starts.pop(0)))
    while mcp_open_starts:
        mcp_spans.append(max(0.0, end_mark - mcp_open_starts.pop(0)))
    if time_to_rag is not None and not rag_spans:
        rag_spans.append(max(0.0, end_mark - (started + time_to_rag)))
    if time_to_mcp is not None and not mcp_spans:
        mcp_spans.append(max(0.0, end_mark - (started + time_to_mcp)))

    data: Any
    if stackchat:
        data = final_payload or {
            "output_text": accumulated,
            "tool_calls": [],
            "raw_output": {"_stream_events": events},
        }
        if accumulated and not data.get("output_text"):
            data["output_text"] = accumulated
        if isinstance(data, dict) and "_stream_events" not in (data.get("raw_output") or {}):
            # Prefer completed message payload; ensure text is present.
            data.setdefault("output_text", accumulated)
    else:
        data = final_payload or {}
        if isinstance(data, dict):
            if accumulated and not _extract_text(data):
                data = {**data, "output_text": accumulated}
            # Merge stream tool events into collectable structure.
            stream_ev = data.pop("_stream_events", None)
            if stream_ev and "output" in data and isinstance(data["output"], list):
                data = {**data, "output": list(data["output"]) + list(stream_ev)}
            elif stream_ev:
                data = {**data, "output": list(stream_ev)}

    result = _finish_trial(
        status=status or 200,
        data=data,
        latency=latency,
        stackchat=stackchat,
    )
    if accumulated and not result.output_text:
        result.output_text = accumulated[:4000]
    # Ensure tool events noticed during SSE are retained.
    if events:
        cleaned = []
        for e in events:
            el = str(e).lower()
            if "delta" in el or "arguments" in el:
                continue
            cleaned.append(e)
        result.tool_events = sorted(set(result.tool_events + cleaned))
    result.mcp_calls = sum(1 for e in result.tool_events if e == "mcp_call")
    result.mcp_list_tools = sum(
        1 for e in result.tool_events if e == "mcp_list_tools"
    )
    result.rag_calls = sum(
        1
        for e in result.tool_events
        if e in {"file_search_call", "retrieve_call"} or "retrieve" in e.lower()
    )
    if rag_spans:
        result.rag_search_spans_s = [round(s, 6) for s in rag_spans]
        result.rag_time_s = sum(rag_spans)
        result.rag_calls = max(result.rag_calls, len(rag_spans))
    if time_to_rag is not None:
        result.time_to_rag_s = time_to_rag
    if mcp_spans:
        result.mcp_call_spans_s = [round(s, 6) for s in mcp_spans]
        result.mcp_time_s = sum(mcp_spans)
        result.mcp_calls = max(result.mcp_calls, len(mcp_spans))
    if time_to_mcp is not None:
        result.time_to_mcp_s = time_to_mcp
    # If file_search ran (seen in final payload) but Stack omitted stream lifecycle
    # events, proxy rag_time as the window before the first MCP call.
    if (
        result.rag_calls
        and result.rag_time_s is None
        and result.time_to_mcp_s is not None
    ):
        result.time_to_rag_s = result.time_to_rag_s or 0.0
        result.rag_time_s = max(0.0, result.time_to_mcp_s - result.time_to_rag_s)
    return _attach_token_metrics(
        result,
        started=started,
        delta_times=delta_times,
        delta_texts=delta_texts,
        usage=usage if usage["output_tokens"] is not None or usage["input_tokens"] is not None else _extract_usage(data),
        streamed=True,
    )


def _normalize_tool_event(name: str) -> str | None:
    """Collapse streaming MCP/RAG event names into stable bench labels."""
    n = (name or "").strip()
    if not n:
        return None
    low = n.lower()
    if "file_search" in low:
        return "file_search_call"
    if "retrieve" in low:
        return "retrieve_call"
    if "mcp_list_tools" in low:
        return "mcp_list_tools"
    if "mcp_call" in low:
        return "mcp_call"
    if low in {"mcp", "tool", "function_call", "file_search_call"}:
        return low
    if "mcp" in low or "tool" in low or "function_call" in low:
        return n
    return None


def _mcp_tool_name_from_event(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("name", "tool_name", "tool"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    params = payload.get("params")
    if isinstance(params, dict):
        name = params.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _is_rag_stream_event(event_name: str, payload: Any) -> bool:
    typ = ""
    if isinstance(payload, dict):
        typ = str(payload.get("type") or "")
    blob = f"{event_name} {typ}".lower()
    if "file_search" in blob:
        return True
    return is_bedrock_rag_tool_name(_mcp_tool_name_from_event(payload))


def _is_mcp_call_stream_event(event_name: str, payload: Any) -> bool:
    typ = ""
    if isinstance(payload, dict):
        typ = str(payload.get("type") or "")
    blob = f"{event_name} {typ}".lower()
    return "mcp_call" in blob and "list_tools" not in blob


def _collect_tool_events(obj: Any, events: list[str] | None = None) -> list[str]:
    events = events if events is not None else []
    if isinstance(obj, dict):
        typ = obj.get("type")
        if isinstance(typ, str):
            norm = _normalize_tool_event(typ)
            if norm:
                events.append(norm)
        for value in obj.values():
            _collect_tool_events(value, events)
    elif isinstance(obj, list):
        for item in obj:
            _collect_tool_events(item, events)
    return events


def _extract_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in {
                "output_text",
                "text",
            }:
                chunks.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                chunks.append(part)
    return "".join(chunks).strip()


def _finish_trial(
    *,
    status: int,
    data: Any,
    latency: float,
    stackchat: bool = False,
) -> TrialResult:
    from benchmarks.phoenix_semantic import (
        extract_jira_issues,
        extract_mcp_tool_responses,
        format_trajectory,
    )

    if not isinstance(data, dict):
        return TrialResult(
            ok=False,
            latency_s=latency,
            status_code=status,
            error="non-json response",
        )

    if stackchat:
        tool_calls = data.get("tool_calls") or []
        events = [
            str(t.get("type") or t.get("name") or "tool")
            for t in tool_calls
            if isinstance(t, dict)
        ]
        events.extend(_collect_tool_events(data.get("raw_output")))
        text = str(data.get("output_text") or "")
        raw = {"tool_calls": tool_calls, "raw_output": data.get("raw_output")}
        response_id = data.get("response_id")
        mcp_source = data.get("raw_output")
    else:
        events = _collect_tool_events(data)
        text = _extract_text(data)
        raw = data
        response_id = data.get("id") or data.get("response_id")
        mcp_source = data

    ok = status == 200 and bool(text or events)
    traj = format_trajectory(raw if not stackchat else data.get("raw_output"), events)
    mcp_responses = extract_mcp_tool_responses(mcp_source)
    jira_issues = extract_jira_issues(mcp_responses)
    return TrialResult(
        ok=ok,
        latency_s=latency,
        status_code=status,
        response_id=response_id,
        output_text=text[:4000],
        tool_events=sorted(set(events)),
        mcp_list_tools=sum(1 for e in events if e == "mcp_list_tools" or "list" in e.lower()),
        mcp_calls=sum(1 for e in events if e == "mcp_call"),
        error=None if ok else json.dumps(data)[:1000],
        raw_output=raw,
        trajectory=traj,
        mcp_tool_responses=mcp_responses,
        jira_issues=jira_issues,
    )


def _build_agent_tools(
    *,
    enable_rag: bool,
    rag_mode: str,
    vector_store_ids: list[str] | None,
    enable_mcp: bool,
    mcp_label: str | None,
    mcp_url: str | None,
    allowed_tools: list[str] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    tools: list[dict[str, Any]] = []
    if enable_rag and rag_mode == "file_search":
        ids = [v for v in (vector_store_ids or []) if v]
        if not ids:
            return tools, "RAG enabled but no vector_store_ids provided"
        tools.append({"type": "file_search", "vector_store_ids": ids})
    if enable_mcp:
        if not mcp_url or not mcp_label:
            return tools, "MCP enabled but mcp_url/mcp_label missing"
        tool: dict[str, Any] = {
            "type": "mcp",
            "server_label": mcp_label,
            "server_url": mcp_url,
            "require_approval": "never",
        }
        if allowed_tools:
            tool["allowed_tools"] = allowed_tools
        tools.append(tool)
    return tools, None


def _run_responses_api(
    *,
    responses_url: str,
    model: str,
    prompt: str,
    instructions: str | None,
    headers: dict[str, str],
    timeout: float,
    stream: bool = True,
    tools: list[dict[str, Any]],
) -> TrialResult:
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "stream": bool(stream),
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if instructions:
        payload["instructions"] = instructions

    if stream:
        try:
            return _stream_request(
                url=responses_url,
                payload=payload,
                headers=headers,
                timeout=timeout,
                stackchat=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Streaming failed ({exc}); falling back to non-stream.")
            payload["stream"] = False

    started = time.perf_counter()
    try:
        status, data = _http_json(
            "POST",
            responses_url,
            payload,
            headers=headers,
            timeout=timeout,
        )
    except ConnectionError as exc:
        return TrialResult(ok=False, latency_s=time.perf_counter() - started, error=str(exc))
    result = _finish_trial(status=status, data=data, latency=time.perf_counter() - started)
    events = result.tool_events
    result.rag_calls = sum(
        1
        for e in events
        if "file_search" in e.lower() or "retrieve" in e.lower()
    )
    return _attach_token_metrics(
        result,
        started=started,
        delta_times=[],
        delta_texts=[],
        usage=_extract_usage(data),
        streamed=False,
    )


def run_llama_stack(
    *,
    base_url: str,
    model: str,
    prompt: str,
    instructions: str | None,
    api_key: str | None,
    timeout: float,
    stream: bool = True,
    enable_mcp: bool = False,
    mcp_label: str | None = None,
    mcp_url: str | None = None,
    allowed_tools: list[str] | None = None,
    enable_rag: bool = False,
    vector_store_ids: list[str] | None = None,
    rag_mode: str = "file_search",
) -> TrialResult:
    tools, err = _build_agent_tools(
        enable_rag=enable_rag,
        rag_mode=rag_mode,
        vector_store_ids=vector_store_ids,
        enable_mcp=enable_mcp,
        mcp_label=mcp_label,
        mcp_url=mcp_url,
        allowed_tools=allowed_tools,
    )
    if err:
        return TrialResult(ok=False, latency_s=0.0, error=err)

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    responses_url = f"{base_url.rstrip('/')}/v1/responses"
    return _run_responses_api(
        responses_url=responses_url,
        model=model,
        prompt=prompt,
        instructions=instructions,
        headers=headers,
        timeout=timeout,
        stream=stream,
        tools=tools,
    )


def run_bedrock_gateway(
    *,
    gateway_url: str,
    model: str,
    prompt: str,
    instructions: str | None,
    timeout: float,
    stream: bool = True,
    enable_mcp: bool = False,
    mcp_label: str | None = None,
    mcp_url: str | None = None,
    allowed_tools: list[str] | None = None,
    enable_rag: bool = False,
    vector_store_ids: list[str] | None = None,
    rag_mode: str = "gateway_mcp",
    region: str | None = None,
    auth_mode: str | None = None,
    api_key: str | None = None,
) -> TrialResult:
    """Agentic bench via Bedrock AgentCore Gateway inference (/inference/v1/responses)."""
    urls = resolve_gateway_urls(gateway_url)
    effective_mcp_url = mcp_url or urls["mcp_url"]
    tools, err = _build_agent_tools(
        enable_rag=enable_rag,
        rag_mode=rag_mode,
        vector_store_ids=vector_store_ids,
        enable_mcp=enable_mcp,
        mcp_label=mcp_label,
        mcp_url=effective_mcp_url,
        allowed_tools=allowed_tools,
    )
    if err:
        return TrialResult(ok=False, latency_s=0.0, error=err)

    payload_preview: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "stream": bool(stream),
    }
    if tools:
        payload_preview["tools"] = tools
    if instructions:
        payload_preview["instructions"] = instructions
    body = json.dumps(payload_preview).encode("utf-8")
    try:
        headers = build_request_headers(
            url=urls["responses_url"],
            method="POST",
            body=body,
            region=region,
            auth_mode=auth_mode,
            bearer_token=api_key,
            extra_headers={
                "Accept": "text/event-stream, application/json",
                "Content-Type": "application/json",
            },
        )
    except RuntimeError as exc:
        return TrialResult(ok=False, latency_s=0.0, error=str(exc))

    return _run_responses_api(
        responses_url=urls["responses_url"],
        model=model,
        prompt=prompt,
        instructions=instructions,
        headers=headers,
        timeout=timeout,
        stream=stream,
        tools=tools,
    )


def run_bedrock_gateway_agent(
    *,
    gateway_url: str,
    prompt: str,
    instructions: str | None,
    timeout: float,
    region: str | None = None,
    auth_mode: str | None = None,
    api_key: str | None = None,
    target: str | None = None,
) -> TrialResult:
    """Invoke AgentCore Runtime agent via Gateway target + Cognito (or bearer/IAM)."""
    try:
        invocation_url = resolve_gateway_target_invocation_url(gateway_url, target)
    except ValueError as exc:
        return TrialResult(ok=False, latency_s=0.0, error=str(exc))

    payload: dict[str, Any] = {"prompt": prompt}
    if instructions:
        payload["instructions"] = instructions
    body = json.dumps(payload).encode("utf-8")

    try:
        headers = build_request_headers(
            url=invocation_url,
            method="POST",
            body=body,
            region=region,
            auth_mode=auth_mode or "cognito",
            bearer_token=api_key,
            extra_headers={
                "Accept": "text/event-stream, application/json",
                "Content-Type": "application/json",
            },
        )
    except RuntimeError as exc:
        return TrialResult(ok=False, latency_s=0.0, error=str(exc))

    started = time.perf_counter()
    req = Request(url=invocation_url, data=body, headers=headers, method="POST")
    accumulated = ""
    delta_times: list[float] = []
    delta_texts: list[str] = []
    deadline = started + timeout
    status = 0
    final_data: dict[str, Any] | None = None

    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310
            status = getattr(resp, "status", 200) or 200
            content_type = (resp.headers.get("Content-Type") or "").lower()

            if "application/json" in content_type and "event-stream" not in content_type:
                raw = resp.read().decode("utf-8", errors="replace")
                try:
                    final_data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    final_data = {"output_text": raw}
                latency = time.perf_counter() - started
                result = _finish_trial(
                    status=status,
                    data=final_data,
                    latency=latency,
                )
                return _attach_token_metrics(
                    result,
                    started=started,
                    delta_times=[],
                    delta_texts=[],
                    usage=_extract_usage(final_data),
                    streamed=False,
                )

            for event_name, data in _iter_sse_events(resp):
                if time.perf_counter() > deadline:
                    break
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                if isinstance(data, dict):
                    text = _extract_text(data)
                    if not text:
                        text = extract_runtime_text(json.dumps(data)) or ""
                    if text:
                        delta_times.append(now)
                        delta_texts.append(text)
                        accumulated += text
                    if data.get("type") in {"response.completed", "response.done"}:
                        final_data = data.get("response") or data
                elif isinstance(data, str):
                    text = extract_runtime_text(data)
                    if text:
                        delta_times.append(now)
                        delta_texts.append(text)
                        accumulated += text
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        latency = time.perf_counter() - started
        try:
            err_data = json.loads(raw) if raw else {"detail": str(exc)}
        except json.JSONDecodeError:
            err_data = {"detail": raw or str(exc)}
        result = _finish_trial(status=exc.code, data=err_data, latency=latency)
        hint = ""
        if "iss" in raw.lower() and "mismatch" in raw.lower():
            hint = (
                " Runtime inbound-auth discoveryUrl must match token iss "
                "(see GW_AGENTIC_COGNITO_DISCOVERY_URL). Gateway ingress auth may "
                "still succeed while the agent runtime behind the target rejects the JWT."
            )
        result.error = (raw[:1000] if raw else str(exc)) + hint
        result.ok = False
        return result
    except URLError as exc:
        return TrialResult(
            ok=False,
            latency_s=time.perf_counter() - started,
            error=str(exc.reason or exc),
        )

    latency = time.perf_counter() - started
    if final_data is None:
        final_data = {"output_text": accumulated, "output": accumulated}
    elif not accumulated and isinstance(final_data, dict):
        accumulated = _extract_text(final_data) or str(
            final_data.get("output_text") or ""
        )

    result = _finish_trial(status=status or 200, data=final_data, latency=latency)
    result.streamed = bool(delta_times)
    return _attach_token_metrics(
        result,
        started=started,
        delta_times=delta_times,
        delta_texts=delta_texts,
        usage=_extract_usage(final_data),
        streamed=bool(delta_times),
    )


def run_bedrock_runtime(
    *,
    agent_runtime_arn: str,
    prompt: str,
    instructions: str | None,
    timeout: float,
    model: str,
    qualifier: str | None = None,
    region: str | None = None,
    runtime_session_id: str | None = None,
    allowed_tools: list[str] | None = None,
    enable_mcp: bool = True,
    auth_mode: str | None = None,
) -> TrialResult:
    """LLM (Converse) + MCP (Runtime InvokeAgentRuntime) tool loop.

    RAG context is expected already inlined in ``prompt`` (bedrock_kb prefetch).
    """
    from benchmarks.phoenix_semantic import extract_jira_issues, format_trajectory

    started = time.perf_counter()
    mcp_url = resolve_runtime_invocations_url(
        agent_runtime_arn, qualifier=qualifier, region=region
    )
    events: list[str] = []
    mcp_responses: list[dict[str, Any]] = []
    mcp_spans: list[float] = []
    time_to_mcp: float | None = None
    mcp_calls = 0
    list_ok = 0
    converse_tools: list[dict[str, Any]] = []
    mcp_client: AgentCoreRuntimeMcpClient | None = None
    server_info = ""

    if enable_mcp:
        mcp_client = AgentCoreRuntimeMcpClient(
            mcp_url,
            auth_mode=auth_mode or "cognito",
            timeout=min(timeout, 120.0),
        )
        try:
            init = mcp_client.initialize()
            if init.get("error"):
                return TrialResult(
                    ok=False,
                    latency_s=time.perf_counter() - started,
                    error=f"Runtime MCP initialize failed: {init.get('error')}",
                )
            info = (init.get("result") or {}).get("serverInfo") or {}
            if isinstance(info, dict):
                server_info = str(info.get("name") or "")
            tools = mcp_client.list_tools()
            list_ok = 1
            events.append("mcp_list_tools")
            converse_tools = mcp_tools_to_converse_specs(tools, allowed_tools)
        except Exception as exc:  # noqa: BLE001
            return TrialResult(
                ok=False,
                latency_s=time.perf_counter() - started,
                error=f"Runtime MCP handshake failed: {exc}",
            )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": prompt}]},
    ]
    accumulated = ""
    delta_times: list[float] = []
    usage_acc = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_stop = ""
    max_rounds = 8

    try:
        for _round in range(max_rounds):
            resp = converse_with_tools(
                model_id=model,
                messages=messages,
                system=instructions,
                tools=converse_tools or None,
                region=region,
            )
            delta_times.append(time.perf_counter())
            usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
            usage_acc["input_tokens"] += int(usage.get("inputTokens") or 0)
            usage_acc["output_tokens"] += int(usage.get("outputTokens") or 0)
            usage_acc["total_tokens"] += int(usage.get("totalTokens") or 0)

            output = resp.get("output") or {}
            message = output.get("message") if isinstance(output, dict) else None
            if not isinstance(message, dict):
                break
            messages.append(message)
            last_stop = str(resp.get("stopReason") or "")

            text_bits: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if isinstance(block.get("text"), str) and block["text"]:
                    text_bits.append(block["text"])
                tool_use = block.get("toolUse")
                if isinstance(tool_use, dict):
                    tool_uses.append(tool_use)
            if text_bits:
                accumulated += "".join(text_bits)

            if last_stop != "tool_use" or not tool_uses or mcp_client is None:
                break

            tool_result_blocks: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                name = str(tool_use.get("name") or "")
                tool_use_id = str(tool_use.get("toolUseId") or "")
                arguments = tool_use.get("input")
                if not isinstance(arguments, dict):
                    arguments = {}
                call_started = time.perf_counter()
                if time_to_mcp is None:
                    time_to_mcp = call_started - started
                mcp_result = mcp_client.call_tool(name, arguments)
                span = time.perf_counter() - call_started
                mcp_spans.append(span)
                mcp_calls += 1
                events.append("mcp_call")
                output_text = mcp_tool_text(mcp_result)
                err = mcp_result.get("error")
                mcp_responses.append(
                    {
                        "type": "mcp_call",
                        "id": tool_use_id,
                        "name": name,
                        "server_label": "jira",
                        "status": "error" if err else "ok",
                        "error": err,
                        "arguments": arguments,
                        "output": output_text,
                    }
                )
                tool_result_blocks.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"text": output_text[:24000]}],
                            "status": "error" if err else "success",
                        }
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})
    except Exception as exc:  # noqa: BLE001
        return TrialResult(
            ok=False,
            latency_s=time.perf_counter() - started,
            error=f"Bedrock Converse / MCP loop failed: {exc}",
            tool_events=sorted(set(events)),
            mcp_calls=mcp_calls,
            mcp_list_tools=list_ok,
            mcp_tool_responses=mcp_responses,
        )

    latency = time.perf_counter() - started
    data: dict[str, Any] = {
        "output_text": accumulated,
        "output": accumulated,
        "stop_reason": last_stop,
        "mcp_calls": mcp_responses,
        "server_info": server_info,
        "usage": {
            "input_tokens": usage_acc["input_tokens"] or None,
            "output_tokens": usage_acc["output_tokens"] or None,
            "total_tokens": usage_acc["total_tokens"] or None,
        },
    }
    ok = bool(accumulated.strip() or mcp_calls)
    traj = format_trajectory(data, events)
    result = TrialResult(
        ok=ok,
        latency_s=latency,
        status_code=200 if ok else None,
        output_text=accumulated[:4000],
        tool_events=sorted(set(events)),
        mcp_list_tools=list_ok,
        mcp_calls=mcp_calls,
        error=None if ok else "empty model output and no MCP calls",
        raw_output=data,
        trajectory=traj,
        mcp_tool_responses=mcp_responses,
        jira_issues=extract_jira_issues(mcp_responses),
        mcp_time_s=sum(mcp_spans) if mcp_spans else None,
        time_to_mcp_s=time_to_mcp,
        mcp_call_spans_s=mcp_spans,
    )
    return _attach_token_metrics(
        result,
        started=started,
        delta_times=delta_times,
        delta_texts=[accumulated] if accumulated else [],
        usage=_extract_usage(data),
        streamed=False,
    )


def run_stackchat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    instructions: str | None,
    timeout: float,
    stream: bool = True,
    enable_mcp: bool = False,
    mcp_label: str | None = None,
    mcp_url: str | None = None,
    allowed_tools: list[str] | None = None,
    enable_rag: bool = False,
    vector_store_ids: list[str] | None = None,
) -> TrialResult:
    payload: dict[str, Any] = {
        "message": prompt,
        "model": model,
        "enable_mcp": bool(enable_mcp),
        "enable_rag": bool(enable_rag),
        "vector_store_ids": list(vector_store_ids or []),
        "stream": bool(stream),
    }
    if enable_mcp and mcp_url and mcp_label:
        server: dict[str, Any] = {
            "server_label": mcp_label,
            "server_url": mcp_url,
        }
        if allowed_tools:
            server["allowed_tools"] = allowed_tools
        payload["mcp_servers"] = [server]
    if instructions:
        payload["instructions"] = instructions

    if stream:
        url = f"{base_url.rstrip('/')}/api/chat/stream"
        try:
            return _stream_request(
                url=url,
                payload=payload,
                headers=None,
                timeout=timeout,
                stackchat=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Streaming failed ({exc}); falling back to non-stream.")
            payload["stream"] = False

    started = time.perf_counter()
    try:
        status, data = _http_json(
            "POST",
            f"{base_url.rstrip('/')}/api/chat",
            payload,
            timeout=timeout,
        )
    except ConnectionError as exc:
        return TrialResult(ok=False, latency_s=time.perf_counter() - started, error=str(exc))
    result = _finish_trial(
        status=status,
        data=data,
        latency=time.perf_counter() - started,
        stackchat=True,
    )
    if enable_rag:
        result.rag_calls = sum(1 for e in result.tool_events if "file_search" in e.lower())
    return _attach_token_metrics(
        result,
        started=started,
        delta_times=[],
        delta_texts=[],
        usage=_extract_usage(data),
        streamed=False,
    )

def _summarize(results: list[TrialResult]) -> dict[str, Any]:
    ordered = sorted(results, key=lambda r: (r.trial_id is None, r.trial_id or 0))
    latencies = [r.latency_s for r in ordered]
    ok_n = sum(1 for r in ordered if r.ok)
    mcp_hit = sum(1 for r in ordered if r.mcp_list_tools or r.mcp_calls or r.tool_events)
    time_stats = _metric_stats(latencies)

    ttfts = [r.ttft_s for r in ordered if r.ttft_s is not None]
    itls = [r.itl_s for r in ordered if r.itl_s is not None]
    in_toks = [float(r.input_tokens) for r in ordered if r.input_tokens is not None]
    out_toks = []
    for r in ordered:
        if r.output_tokens is not None:
            out_toks.append(float(r.output_tokens))
        elif r.output_tokens_estimated:
            out_toks.append(float(r.output_tokens_estimated))
    total_toks = [float(r.total_tokens) for r in ordered if r.total_tokens is not None]
    tps = [r.tokens_per_second for r in ordered if r.tokens_per_second is not None]
    rag_times = [r.rag_time_s for r in ordered if r.rag_time_s is not None]
    time_to_rags = [r.time_to_rag_s for r in ordered if r.time_to_rag_s is not None]
    rag_calls_total = sum(r.rag_calls for r in ordered)
    mcp_times = [r.mcp_time_s for r in ordered if r.mcp_time_s is not None]
    time_to_mcps = [r.time_to_mcp_s for r in ordered if r.time_to_mcp_s is not None]
    mcp_calls_total = sum(r.mcp_calls for r in ordered)

    per_request = [
        {
            "trial_id": r.trial_id if r.trial_id is not None else i,
            "ok": r.ok,
            "total_time_s": round(r.total_time_s, 4),
            "orchestration_time_s": round(r.latency_s, 4),
            "ttft_s": round(r.ttft_s, 4) if r.ttft_s is not None else None,
            "itl_s": round(r.itl_s, 6) if r.itl_s is not None else None,
            "rag_time_s": round(r.rag_time_s, 4) if r.rag_time_s is not None else None,
            "time_to_rag_s": (
                round(r.time_to_rag_s, 4) if r.time_to_rag_s is not None else None
            ),
            "rag_calls": r.rag_calls,
            "mcp_time_s": round(r.mcp_time_s, 4) if r.mcp_time_s is not None else None,
            "time_to_mcp_s": (
                round(r.time_to_mcp_s, 4) if r.time_to_mcp_s is not None else None
            ),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "output_tokens_estimated": r.output_tokens_estimated,
            "total_tokens": r.total_tokens,
            "tokens_per_second": (
                round(r.tokens_per_second, 4) if r.tokens_per_second is not None else None
            ),
            "streamed": r.streamed,
            "stream_delta_count": r.stream_delta_count,
            "mcp_calls": r.mcp_calls,
            "tool_events": r.tool_events,
        }
        for i, r in enumerate(ordered)
    ]
    return {
        "requests": len(ordered),
        "ok": ok_n,
        "errors": len(ordered) - ok_n,
        "success_rate": (ok_n / len(ordered)) if ordered else 0.0,
        "tool_activity_rate": (mcp_hit / len(ordered)) if ordered else 0.0,
        "total_time_s": time_stats,
        "orchestration_time_s": time_stats,
        "latency_s": time_stats,
        "ttft_s": _metric_stats(ttfts),
        "itl_s": _metric_stats(itls),
        "rag_time_s": _metric_stats(rag_times),
        "time_to_rag_s": _metric_stats(time_to_rags),
        "rag_calls_total": rag_calls_total,
        "mcp_time_s": _metric_stats(mcp_times),
        "time_to_mcp_s": _metric_stats(time_to_mcps),
        "mcp_calls_total": mcp_calls_total,
        "input_tokens": _metric_stats(in_toks),
        "output_tokens": _metric_stats(out_toks),
        "total_tokens": _metric_stats(total_toks),
        "tokens_per_second": _metric_stats(tps),
        "streamed_requests": sum(1 for r in ordered if r.streamed),
        "per_request": per_request,
    }


def _trial_dict(r: TrialResult) -> dict[str, Any]:
    d = asdict(r)
    d.pop("raw_output", None)
    # Keep compact issue view in main report; full MCP payloads go to jira-responses file.
    d["jira_issues"] = [
        {k: v for k, v in issue.items() if k != "raw"} for issue in (r.jira_issues or [])
    ]
    d["mcp_tool_responses"] = [
        {
            "name": c.get("name"),
            "status": c.get("status"),
            "error": c.get("error"),
            "arguments": c.get("arguments"),
            # truncate huge payloads in the main report
            "output_preview": _preview_output(c.get("output")),
        }
        for c in (r.mcp_tool_responses or [])
    ]
    d["total_time_s"] = round(r.total_time_s, 4)
    d["orchestration_time_s"] = round(r.latency_s, 4)
    if r.ttft_s is not None:
        d["ttft_s"] = round(r.ttft_s, 4)
    if r.itl_s is not None:
        d["itl_s"] = round(r.itl_s, 6)
    if r.itl_s_median is not None:
        d["itl_s_median"] = round(r.itl_s_median, 6)
    if r.itl_s_p95 is not None:
        d["itl_s_p95"] = round(r.itl_s_p95, 6)
    if r.tokens_per_second is not None:
        d["tokens_per_second"] = round(r.tokens_per_second, 4)
    if r.rag_time_s is not None:
        d["rag_time_s"] = round(r.rag_time_s, 4)
    if r.time_to_rag_s is not None:
        d["time_to_rag_s"] = round(r.time_to_rag_s, 4)
    if r.mcp_time_s is not None:
        d["mcp_time_s"] = round(r.mcp_time_s, 4)
    if r.time_to_mcp_s is not None:
        d["time_to_mcp_s"] = round(r.time_to_mcp_s, 4)
    return d


def _preview_output(output: Any, limit: int = 500) -> Any:
    if output is None:
        return None
    if isinstance(output, (dict, list)):
        text = json.dumps(output, ensure_ascii=False)
        return text if len(text) <= limit else text[:limit] + "…"
    text = str(output)
    return text if len(text) <= limit else text[:limit] + "…"


def write_jira_responses_file(
    path: Path,
    *,
    stamp: str,
    prompt: str,
    model: str,
    mcp: dict[str, Any],
    results: list[TrialResult],
) -> Path:
    """Write a dedicated per-bench file with raw Jira/MCP tool responses."""
    ordered = sorted(results, key=lambda r: r.trial_id or 0)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": stamp,
        "prompt": prompt,
        "model": model,
        "mcp": mcp,
        "requests": [
            {
                "trial_id": r.trial_id,
                "ok": r.ok,
                "total_time_s": round(r.total_time_s, 4),
                "ttft_s": round(r.ttft_s, 4) if r.ttft_s is not None else None,
                "itl_s": round(r.itl_s, 6) if r.itl_s is not None else None,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "output_tokens_estimated": r.output_tokens_estimated,
                "total_tokens": r.total_tokens,
                "tokens_per_second": (
                    round(r.tokens_per_second, 4) if r.tokens_per_second is not None else None
                ),
                "streamed": r.streamed,
                "response_id": r.response_id,
                "assistant_summary": r.output_text,
                "jira_issues": r.jira_issues,
                "mcp_tool_responses": r.mcp_tool_responses,
                "trajectory": r.trajectory,
            }
            for r in ordered
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def write_rag_responses_file(
    path: Path,
    *,
    stamp: str,
    prompt: str,
    model: str,
    rag: dict[str, Any],
    results: list[TrialResult],
) -> Path:
    """Write per-bench RAG timing + assistant answers."""
    ordered = sorted(results, key=lambda r: r.trial_id or 0)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": stamp,
        "prompt": prompt,
        "model": model,
        "rag": rag,
        "requests": [
            {
                "trial_id": r.trial_id,
                "ok": r.ok,
                "total_time_s": round(r.total_time_s, 4),
                "ttft_s": round(r.ttft_s, 4) if r.ttft_s is not None else None,
                "rag_time_s": round(r.rag_time_s, 4) if r.rag_time_s is not None else None,
                "time_to_rag_s": (
                    round(r.time_to_rag_s, 4) if r.time_to_rag_s is not None else None
                ),
                "rag_calls": r.rag_calls,
                "rag_search_spans_s": r.rag_search_spans_s,
                "mcp_time_s": round(r.mcp_time_s, 4) if r.mcp_time_s is not None else None,
                "time_to_mcp_s": (
                    round(r.time_to_mcp_s, 4) if r.time_to_mcp_s is not None else None
                ),
                "mcp_calls": r.mcp_calls,
                "mcp_call_spans_s": r.mcp_call_spans_s,
                "itl_s": round(r.itl_s, 6) if r.itl_s is not None else None,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "total_tokens": r.total_tokens,
                "tokens_per_second": (
                    round(r.tokens_per_second, 4) if r.tokens_per_second is not None else None
                ),
                "streamed": r.streamed,
                "response_id": r.response_id,
                "assistant_summary": r.output_text,
                "tool_events": r.tool_events,
                "trajectory": r.trajectory,
            }
            for r in ordered
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def _print_final_report(summary: dict[str, Any], suite_wall_s: float) -> None:
    print()
    print("=== Benchmark final report ===")
    print(
        f"requests={summary.get('requests')}  ok={summary.get('ok')}  "
        f"errors={summary.get('errors')}  "
        f"success_rate={summary.get('success_rate', 0):.0%}"
    )
    if "combined_success_rate" in summary:
        print(
            f"semantic={summary.get('semantic_success_rate', 0):.0%}  "
            f"trajectory={summary.get('trajectory_success_rate', 0):.0%}  "
            f"combined={summary.get('combined_success_rate', 0):.0%}  "
            f"judge={summary.get('judge_model')}"
        )
    totals = summary.get("total_time_s") or {}
    print(
        "total_time_s "
        f"min={totals.get('min')}  max={totals.get('max')}  "
        f"mean={totals.get('mean')}  sum={totals.get('sum')}  "
        f"suite_wall={round(suite_wall_s, 4)}"
    )
    ttft = summary.get("ttft_s") or {}
    itl = summary.get("itl_s") or {}
    out_tok = summary.get("output_tokens") or {}
    in_tok = summary.get("input_tokens") or {}
    print(
        "ttft_s "
        f"mean={ttft.get('mean')}  median={ttft.get('median')}  p95={ttft.get('p95')}"
    )
    print(
        "itl_s "
        f"mean={itl.get('mean')}  median={itl.get('median')}  p95={itl.get('p95')}"
    )
    rag = summary.get("rag_time_s") or {}
    ttrag = summary.get("time_to_rag_s") or {}
    if rag.get("mean") is not None or summary.get("rag_calls_total"):
        print(
            "rag_time_s "
            f"mean={rag.get('mean')}  median={rag.get('median')}  p95={rag.get('p95')}  "
            f"sum={rag.get('sum')}  calls={summary.get('rag_calls_total')}  "
            f"time_to_rag_mean={ttrag.get('mean')}"
        )
    mcp_t = summary.get("mcp_time_s") or {}
    ttmcp = summary.get("time_to_mcp_s") or {}
    if mcp_t.get("mean") is not None or summary.get("mcp_calls_total"):
        print(
            "mcp_time_s "
            f"mean={mcp_t.get('mean')}  median={mcp_t.get('median')}  p95={mcp_t.get('p95')}  "
            f"sum={mcp_t.get('sum')}  calls={summary.get('mcp_calls_total')}  "
            f"time_to_mcp_mean={ttmcp.get('mean')}"
        )
    print(
        "tokens "
        f"input_mean={in_tok.get('mean')}  output_mean={out_tok.get('mean')}  "
        f"output_sum={out_tok.get('sum')}  "
        f"tps_mean={(summary.get('tokens_per_second') or {}).get('mean')}"
    )
    print()
    print("per-request latency / tokens:")
    print(
        f"{'req':>4}  {'ok':>3}  {'total_s':>8}  {'ttft_s':>8}  {'rag_s':>8}  "
        f"{'mcp_s':>8}  {'itl_s':>10}  {'out_tok':>7}  tools"
    )
    for row in summary.get("per_request") or []:
        tools = ",".join(row.get("tool_events") or []) or "-"
        out = row.get("output_tokens")
        if out is None:
            out = row.get("output_tokens_estimated")
        ttft_v = row.get("ttft_s")
        itl_v = row.get("itl_s")
        rag_v = row.get("rag_time_s")
        mcp_v = row.get("mcp_time_s")
        print(
            f"{row.get('trial_id'):>4}  "
            f"{'Y' if row.get('ok') else 'N':>3}  "
            f"{float(row.get('total_time_s') or 0):>8.4f}  "
            f"{(f'{ttft_v:.4f}' if ttft_v is not None else '-'):>8}  "
            f"{(f'{rag_v:.4f}' if rag_v is not None else '-'):>8}  "
            f"{(f'{mcp_v:.4f}' if mcp_v is not None else '-'):>8}  "
            f"{(f'{itl_v:.6f}' if itl_v is not None else '-'):>10}  "
            f"{(str(out) if out is not None else '-'):>7}  {tools}"
        )
    print()
    print(json.dumps(summary, indent=2))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(root / "backend" / ".env")
    _load_dotenv(root / "backend" / ".env.bedrock", override=True)
    profile_id = (os.getenv("BENCH_MODEL_INFERENCE_PROFILE_ID") or "").strip()
    if profile_id:
        os.environ["BENCH_MODEL"] = profile_id

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("llama-stack", "stackchat", "bedrock-gateway", "bedrock-runtime"),
        default=os.getenv("BENCH_AGENTIC_TARGET", "llama-stack"),
    )
    parser.add_argument(
        "--bedrock-gateway-url",
        default=os.getenv("BEDROCK_GATEWAY_URL") or "",
        help="AgentCore Gateway base URL (for --target bedrock-gateway)",
    )
    parser.add_argument(
        "--bedrock-runtime-arn",
        default=os.getenv("BEDROCK_AGENT_RUNTIME_ARN")
        or os.getenv("GW_AGENTIC_RUNTIME_ARN")
        or "",
        help="AgentCore Runtime ARN (MCP server for --target bedrock-runtime)",
    )
    parser.add_argument(
        "--bedrock-runtime-qualifier",
        default=os.getenv("BEDROCK_RUNTIME_QUALIFIER") or "",
        help="Optional AgentCore Runtime endpoint qualifier",
    )
    parser.add_argument(
        "--bedrock-region",
        default=os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or "",
        help="AWS region for SigV4 / boto3 (defaults from AWS_REGION)",
    )
    parser.add_argument(
        "--bedrock-auth",
        choices=("iam", "bearer", "cognito", "api-key"),
        default=resolve_auth_mode(os.getenv("BEDROCK_AUTH_MODE")),
        help="Gateway auth: cognito (agent target), iam, bearer, or api-key",
    )
    parser.add_argument(
        "--gateway-inference-only",
        action="store_true",
        default=os.getenv("BENCH_GATEWAY_INFERENCE_ONLY", "").strip().lower()
        in {"1", "true", "yes"},
        help="Use /inference/v1/responses instead of gateway runtime target invocations",
    )
    parser.add_argument(
        "--rag-mode",
        choices=("file_search", "gateway_mcp", "bedrock_kb"),
        default=(os.getenv("BENCH_RAG_MODE") or "").strip().lower() or None,
        help="RAG: file_search (Llama Stack), bedrock_kb (Retrieve API), or gateway_mcp (optional KB connector on /mcp)",
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--model",
        default=os.getenv("BENCH_MODEL")
        or os.getenv("BENCH_MODEL_INFERENCE_PROFILE_ID")
        or os.getenv("DEFAULT_MODEL")
        or "vllm-inference-1/llama-32-fp8",
    )
    parser.add_argument(
        "--scenario",
        choices=("list", "create", "rag"),
        default=os.getenv("BENCH_SCENARIO", "list"),
        help="list/create = Jira MCP; rag = file_search vector store retrieval",
    )
    parser.add_argument("--prompt", default=os.getenv("BENCH_PROMPT") or None)
    parser.add_argument(
        "--system-prompt-file",
        default=os.getenv("BENCH_SYSTEM_PROMPT_FILE") or None,
        help="System/instructions prompt file",
    )
    parser.add_argument(
        "--user-prompt-file",
        default=os.getenv("BENCH_USER_PROMPT_FILE") or None,
        help="User prompt file filled by the operator before the run",
    )
    parser.add_argument(
        "--vector-store-ids",
        default=os.getenv("BENCH_VECTOR_STORE_IDS")
        or os.getenv("DEFAULT_VECTOR_STORE_IDS")
        or "",
        help="Comma-separated vector store ids (required for --scenario rag)",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("BENCH_MCP_SERVER_URL") or os.getenv("MCP_SERVER_URL"),
    )
    parser.add_argument(
        "--mcp-label",
        default=os.getenv("BENCH_MCP_SERVER_LABEL", "jira"),
    )
    parser.add_argument(
        "--knowledge-base-id",
        default=os.getenv("BEDROCK_KNOWLEDGE_BASE_ID")
        or os.getenv("GW_KNOWLEDGE_BASE_QUICK_START")
        or "",
        help="Bedrock managed KB id (10-char) for bedrock_kb Retrieve API",
    )
    parser.add_argument(
        "--kb-embedding-model",
        default=os.getenv("BEDROCK_KB_EMBEDDING_MODEL")
        or os.getenv("GW_KB_EMBEDDING_MODEL")
        or "",
        help="KB embedding model id (e.g. amazon.titan-embed-text-v2:0) — metadata only",
    )
    parser.add_argument(
        "--allowed-tools",
        default=os.getenv("BENCH_MCP_ALLOWED_TOOLS", ""),
    )
    parser.add_argument(
        "--instructions",
        default=os.getenv("BENCH_INSTRUCTIONS") or None,
    )
    parser.add_argument("--requests", type=int, default=int(os.getenv("BENCH_REQUESTS", "5")))
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("BENCH_CONCURRENCY", "1")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("BENCH_TIMEOUT", "180")),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("BENCH_API_KEY") or os.getenv("LLAMA_STACK_API_KEY") or "",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--phoenix",
        action="store_true",
        default=os.getenv("BENCH_PHOENIX", "").lower() in {"1", "true", "yes"},
        help="Run Phoenix semantic + trajectory judges and export traces",
    )
    parser.add_argument(
        "--no-phoenix",
        action="store_true",
        help="Disable Phoenix even if BENCH_PHOENIX is set",
    )
    parser.add_argument(
        "--phoenix-project",
        default=os.getenv("PHOENIX_PROJECT_NAME", "jira-mcp-agentic"),
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("PHOENIX_JUDGE_MODEL") or os.getenv("BENCH_JUDGE_MODEL") or "",
        help=(
            "Phoenix LLM-as-judge model id (Amazon Bedrock inference profile by default; "
            "set PHOENIX_JUDGE_PROVIDER=llama-stack for an OpenAI-compatible /v1 judge)"
        ),
    )
    parser.add_argument(
        "--skip-trajectory-gate",
        action="store_true",
        help="Combined success only requires semantic correct (not trajectory)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=os.getenv("BENCH_STREAM", "1").lower() in {"0", "false", "no"},
        help="Disable streaming (TTFT/ITL unavailable; tokens still read from usage if present)",
    )
    args = parser.parse_args()
    if args.gateway_inference_only:
        os.environ["BENCH_GATEWAY_INFERENCE_ONLY"] = "1"
    if args.bedrock_runtime_arn.strip():
        os.environ.setdefault("BEDROCK_AGENT_RUNTIME_ARN", args.bedrock_runtime_arn.strip())
    elif not args.bedrock_runtime_arn.strip():
        inferred = resolve_agent_runtime_arn() or ""
        if inferred:
            args.bedrock_runtime_arn = inferred
    if args.knowledge_base_id.strip():
        os.environ["BEDROCK_KNOWLEDGE_BASE_ID"] = args.knowledge_base_id.strip()
    if args.kb_embedding_model.strip():
        os.environ["BEDROCK_KB_EMBEDDING_MODEL"] = args.kb_embedding_model.strip()
    use_phoenix = args.phoenix and not args.no_phoenix
    use_stream = not args.no_stream
    scenario = args.scenario

    try:
        if scenario == "create":
            system_path = Path(
                args.system_prompt_file or (root / DEFAULT_CREATE_SYSTEM)
            )
            user_path = Path(args.user_prompt_file or (root / DEFAULT_CREATE_USER))
            if not system_path.is_absolute():
                system_path = (root / system_path).resolve()
            if not user_path.is_absolute():
                user_path = (root / user_path).resolve()
            instructions = args.instructions or load_prompt_file(
                system_path, require_filled=False
            )
            prompt = args.prompt or load_prompt_file(user_path, require_filled=True)
            prompt_meta = {
                "system_prompt_file": str(system_path),
                "user_prompt_file": str(user_path),
            }
        elif scenario == "rag":
            system_path = Path(args.system_prompt_file or (root / DEFAULT_RAG_SYSTEM))
            user_path = Path(args.user_prompt_file or (root / DEFAULT_RAG_USER))
            if not system_path.is_absolute():
                system_path = (root / system_path).resolve()
            if not user_path.is_absolute():
                user_path = (root / user_path).resolve()
            instructions = args.instructions or (
                load_prompt_file(system_path, require_filled=False)
                if system_path.is_file()
                else (
                    "You are an assistant with RAG and Jira MCP. "
                    "Use file_search for company procedures, then MCP for Jira actions. "
                    "Never invent issue keys."
                )
            )
            if args.prompt:
                prompt = args.prompt
            elif user_path.is_file():
                prompt = load_prompt_file(user_path, require_filled=True)
            else:
                prompt = DEFAULT_RAG_PROMPT
            prompt_meta = {
                "system_prompt_file": str(system_path) if system_path.is_file() else None,
                "user_prompt_file": str(user_path) if user_path.is_file() else None,
            }
        else:
            if args.system_prompt_file:
                sp = Path(args.system_prompt_file)
                if not sp.is_absolute():
                    sp = (root / sp).resolve()
                instructions = args.instructions or load_prompt_file(
                    sp, require_filled=False
                )
                prompt_meta = {"system_prompt_file": str(sp)}
            else:
                instructions = args.instructions or DEFAULT_LIST_INSTRUCTIONS
                prompt_meta = {}
            if args.user_prompt_file:
                up = Path(args.user_prompt_file)
                if not up.is_absolute():
                    up = (root / up).resolve()
                prompt = args.prompt or load_prompt_file(up, require_filled=True)
                prompt_meta["user_prompt_file"] = str(up)
            else:
                prompt = args.prompt or DEFAULT_PROMPT
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    vector_store_ids = [
        x.strip() for x in str(args.vector_store_ids or "").split(",") if x.strip()
    ]
    enable_rag = scenario == "rag"
    enable_mcp = scenario in {"list", "create", "rag"}

    if args.rag_mode:
        rag_mode = args.rag_mode
    elif args.target.startswith("bedrock"):
        rag_mode = "bedrock_kb"
    else:
        rag_mode = "file_search"

    gateway_url = (
        resolve_bedrock_gateway_url(args.bedrock_gateway_url or args.base_url)
    ).strip()
    if args.target.startswith("bedrock") and not args.mcp_url:
        if args.target == "bedrock-runtime" and (args.bedrock_runtime_arn or "").strip():
            args.mcp_url = resolve_runtime_invocations_url(
                args.bedrock_runtime_arn.strip(),
                qualifier=args.bedrock_runtime_qualifier or None,
                region=args.bedrock_region or None,
            )
        elif gateway_url:
            args.mcp_url = resolve_gateway_urls(gateway_url)["mcp_url"]

    if enable_rag and rag_mode == "file_search" and not vector_store_ids:
        print(
            "Missing vector store ids for file_search RAG. Pass --vector-store-ids or set "
            "BENCH_VECTOR_STORE_IDS / DEFAULT_VECTOR_STORE_IDS.",
            file=sys.stderr,
        )
        return 2

    agent_gateway_target = (
        args.target == "bedrock-gateway" and resolve_gateway_target_name()
    )

    if enable_mcp and not args.mcp_url and args.target != "bedrock-runtime" and not agent_gateway_target:
        print(
            "Missing MCP URL. Set BENCH_MCP_SERVER_URL, BEDROCK_GATEWAY_URL, or --mcp-url.\n"
            "For Llama Stack, the URL must be reachable from the Stack host.",
            file=sys.stderr,
        )
        return 2

    if args.target == "bedrock-gateway" and not gateway_url:
        print(
            "Missing AgentCore Gateway URL. Set BEDROCK_GATEWAY_URL or --bedrock-gateway-url.",
            file=sys.stderr,
        )
        return 2

    if args.target == "bedrock-runtime" and not args.bedrock_runtime_arn.strip():
        print(
            "Missing AgentCore Runtime ARN. Set BEDROCK_AGENT_RUNTIME_ARN or --bedrock-runtime-arn.",
            file=sys.stderr,
        )
        return 2

    if enable_rag and rag_mode == "bedrock_kb":
        kb_api = resolve_knowledge_base_api_id(region=args.bedrock_region or None)
        if not kb_api:
            print(
                "Missing Bedrock knowledge base id for bedrock_kb RAG. Set "
                "BEDROCK_KNOWLEDGE_BASE_ID (10-char) or GW_KNOWLEDGE_BASE_QUICK_START.",
                file=sys.stderr,
            )
            return 2

    if enable_rag and rag_mode == "gateway_mcp":
        kb_id = resolve_knowledge_base_id() or args.knowledge_base_id.strip() or None
        kb_tools = resolve_kb_gateway_mcp_tool_names()
        retrieve_names = ", ".join(kb_tools[:2])
        if kb_id and "retrieve" not in (instructions or "").lower():
            instructions = (
                (instructions or "").strip()
                + f"\n\nUse AgentCore Gateway MCP tools ({retrieve_names}) to query "
                f"the managed knowledge base '{kb_id}' before Jira MCP actions. "
                "Pass retrievalQuery.text with the user's question. Never invent facts or issue keys."
            ).strip()
        elif "retrieve" not in (instructions or "").lower():
            instructions = (
                (instructions or "").strip()
                + "\n\nUse AgentCore Gateway MCP tools Retrieve or AgenticRetrieveStream to query "
                "the knowledge base before Jira MCP actions. Never invent issue keys."
            ).strip()

    if enable_rag and rag_mode == "bedrock_kb" and "knowledge base" not in (instructions or "").lower():
        kb_label = resolve_knowledge_base_name() or resolve_knowledge_base_api_id() or "configured KB"
        instructions = (
            (instructions or "").strip()
            + f"\n\nAnswer using the retrieved Bedrock Knowledge Base context ('{kb_label}'). "
            "Cite facts from the passages in the user message. "
            "If nothing relevant was retrieved, say so. Never invent procedures or issue keys."
        ).strip()

    if args.target == "llama-stack":
        base_url = args.base_url or os.getenv("LLAMA_STACK_BASE_URL") or "http://localhost:8321"
    elif args.target == "bedrock-gateway":
        base_url = gateway_url
    elif args.target == "bedrock-runtime":
        base_url = args.bedrock_runtime_arn
    else:
        base_url = (
            args.base_url
            or os.getenv("STACKCHAT_BASE_URL")
            or f"http://127.0.0.1:{os.getenv('PORT', '8000')}"
        )

    allowed = [t.strip() for t in args.allowed_tools.split(",") if t.strip()] or None
    if enable_rag and rag_mode == "gateway_mcp":
        kb_tool_names = resolve_kb_gateway_mcp_tool_names()
        if allowed:
            allowed = list(dict.fromkeys(allowed + kb_tool_names))
        else:
            allowed = kb_tool_names
    results_dir = root / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if scenario == "create":
        prefix = "agentic-jira-create"
    elif scenario == "rag":
        prefix = "agentic-rag"
    else:
        prefix = "agentic-jira"
    if args.target.startswith("bedrock"):
        prefix = f"agentic-bedrock-{prefix.replace('agentic-', '')}"
    out_path = Path(args.output) if args.output else results_dir / f"{prefix}-{stamp}.json"
    jira_path = results_dir / f"{prefix}-{stamp}-jira-responses.json"
    rag_path = results_dir / f"{prefix}-{stamp}-rag-responses.json"

    tools_payload: list[dict[str, Any]] = []
    if enable_rag and rag_mode == "file_search":
        tools_payload.append(
            {"type": "file_search", "vector_store_ids": vector_store_ids}
        )
    if enable_rag and rag_mode == "bedrock_kb":
        tools_payload.append(
            {
                "type": "bedrock_kb_retrieve",
                "knowledge_base_id": resolve_knowledge_base_api_id(),
                "knowledge_base_name": resolve_knowledge_base_name(),
                "kb_embedding_model": resolve_kb_embedding_model(),
                "api": "bedrock-agent-runtime:Retrieve",
            }
        )
    if enable_rag and rag_mode == "gateway_mcp":
        tools_payload.append(
            {
                "type": "gateway_mcp_rag",
                "mcp_tools": resolve_kb_gateway_mcp_tool_names(),
                "knowledge_base_id": resolve_knowledge_base_api_id(),
                "kb_embedding_model": resolve_kb_embedding_model(),
                "gateway_mcp_url": args.mcp_url,
            }
        )
    if enable_mcp and args.target != "bedrock-runtime":
        tools_payload.append(
            {
                "type": "mcp",
                "server_label": args.mcp_label,
                "server_url": args.mcp_url,
                "allowed_tools": allowed,
            }
        )
    if enable_mcp and args.target == "bedrock-runtime":
        tools_payload.append(
            {
                "type": "mcp",
                "server_label": args.mcp_label,
                "server_url": args.mcp_url,
                "transport": "agentcore-runtime",
                "allowed_tools": allowed,
            }
        )
    tools_desc = json.dumps(tools_payload, indent=2)

    if use_phoenix:
        from benchmarks.phoenix_semantic import configure_phoenix_workdir, ensure_phoenix_tracing

        configure_phoenix_workdir(root / "benchmarks")
        ensure_phoenix_tracing(args.phoenix_project)

    print(f"Target:       {args.target}")
    print(f"Scenario:     {scenario}")
    print(f"Base URL:     {base_url}")
    print(f"Model:        {args.model}")
    if enable_rag:
        print(f"RAG mode:     {rag_mode}")
        if rag_mode == "file_search" and vector_store_ids:
            print(f"RAG stores:   {', '.join(vector_store_ids)}")
        if rag_mode == "bedrock_kb":
            print(f"RAG KB:       {resolve_knowledge_base_name() or '-'} ({resolve_knowledge_base_api_id() or '-'})")
            emb = resolve_kb_embedding_model()
            if emb:
                print(f"RAG embed:    {emb}")
            print("RAG API:      bedrock-agent-runtime Retrieve")
        if rag_mode == "gateway_mcp":
            kb_id = resolve_knowledge_base_id()
            if kb_id:
                print(f"RAG KB:       {kb_id}")
            emb = resolve_kb_embedding_model()
            if emb:
                print(f"RAG embed:    {emb}")
            print(f"RAG MCP tools:{', '.join(resolve_kb_gateway_mcp_tool_names()[:2])}")
    if enable_mcp:
        print(f"MCP:          {args.mcp_label} → {args.mcp_url}")
    if args.target.startswith("bedrock"):
        print(f"Bedrock auth: {args.bedrock_auth}")
        if args.bedrock_region:
            print(f"Bedrock region: {args.bedrock_region}")
    for key, value in prompt_meta.items():
        if value:
            print(f"{key}: {value}")
    print(f"Prompt:       {prompt}")
    print(f"Requests:     {args.requests} (concurrency={args.concurrency})")
    print(f"Stream:       {'on' if use_stream else 'off'} (TTFT/ITL/RAG)")
    print(f"Phoenix:      {'on' if use_phoenix else 'off'}")
    print()

    if scenario == "create":
        span_name = "agentic.jira_create_issue"
    elif scenario == "rag":
        span_name = (
            "agentic.bedrock_rag_mcp"
            if args.target.startswith("bedrock")
            else "agentic.rag_file_search"
        )
    else:
        span_name = "agentic.jira_list_sup"

    def one(idx: int) -> TrialResult:
        trial_prompt = prompt
        rag_prefetch_s: float | None = None

        if enable_rag and rag_mode == "bedrock_kb" and args.target.startswith("bedrock"):
            rag_started = time.perf_counter()
            try:
                chunks = retrieve_knowledge_base(
                    prompt,
                    region=args.bedrock_region or None,
                )
                rag_prefetch_s = time.perf_counter() - rag_started
                trial_prompt = (
                    format_kb_retrieval_context(chunks)
                    + "\n\nQuestion: "
                    + prompt
                )
            except Exception as exc:  # noqa: BLE001
                result = TrialResult(
                    ok=False,
                    latency_s=0.0,
                    error=f"Bedrock KB Retrieve failed: {exc}",
                    trial_id=idx,
                )
                return result

        if args.target == "bedrock-runtime":
            result = run_bedrock_runtime(
                agent_runtime_arn=args.bedrock_runtime_arn,
                prompt=trial_prompt,
                instructions=instructions,
                timeout=args.timeout,
                model=args.model,
                qualifier=args.bedrock_runtime_qualifier or None,
                region=args.bedrock_region or None,
                allowed_tools=allowed,
                enable_mcp=enable_mcp,
                auth_mode=args.bedrock_auth,
            )
        else:
            common = dict(
                base_url=base_url,
                model=args.model,
                prompt=trial_prompt,
                instructions=instructions,
                timeout=args.timeout,
                stream=use_stream,
                enable_mcp=enable_mcp,
                mcp_label=args.mcp_label,
                mcp_url=args.mcp_url,
                allowed_tools=allowed,
                enable_rag=enable_rag,
                vector_store_ids=vector_store_ids,
                rag_mode=rag_mode,
            )
            if args.target == "llama-stack":
                result = run_llama_stack(
                    api_key=args.api_key or None,
                    **common,
                )
            elif args.target == "bedrock-gateway":
                agent_target = resolve_gateway_target_name()
                if agent_target:
                    result = run_bedrock_gateway_agent(
                        gateway_url=gateway_url,
                        prompt=trial_prompt,
                        instructions=instructions,
                        timeout=args.timeout,
                        region=args.bedrock_region or None,
                        auth_mode="cognito",
                        api_key=args.api_key or None,
                        target=agent_target,
                    )
                else:
                    result = run_bedrock_gateway(
                        gateway_url=gateway_url,
                        region=args.bedrock_region or None,
                        auth_mode=args.bedrock_auth,
                        api_key=args.api_key or None,
                        **{k: v for k, v in common.items() if k != "base_url"},
                    )
            else:
                result = run_stackchat(**common)
        if rag_prefetch_s is not None:
            result.rag_time_s = rag_prefetch_s
            result.time_to_rag_s = 0.0
            result.rag_calls = max(result.rag_calls, 1)
        result.trial_id = idx
        if use_phoenix:
            from benchmarks.phoenix_semantic import trace_agent_span

            trace_agent_span(
                name=span_name,
                prompt=prompt,
                output_text=result.output_text,
                trajectory=result.trajectory,
                orchestration_time_s=result.latency_s,
                attributes={
                    "trial_id": idx,
                    "scenario": scenario,
                    "total_time_s": result.total_time_s,
                    "rag_time_s": result.rag_time_s,
                    "time_to_rag_s": result.time_to_rag_s,
                    "rag_calls": result.rag_calls,
                    "mcp_time_s": result.mcp_time_s,
                    "time_to_mcp_s": result.time_to_mcp_s,
                    "response_id": result.response_id,
                    "http_ok": result.ok,
                    "mcp_list_tools": result.mcp_list_tools,
                    "mcp_calls": result.mcp_calls,
                    "tool_events": result.tool_events,
                },
            )
        return result

    results: list[TrialResult] = []
    suite_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(one, i): i for i in range(args.requests)}
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            flag = "OK" if result.ok else "ERR"
            tools = ",".join(result.tool_events) or "-"
            print(
                f"[{flag}] req={result.trial_id}  "
                f"total_time={result.total_time_s:6.2f}s  "
                f"ttft={result.ttft_s if result.ttft_s is not None else '-'}  "
                f"rag={result.rag_time_s if result.rag_time_s is not None else '-'}  "
                f"mcp={result.mcp_time_s if result.mcp_time_s is not None else '-'}  "
                f"itl={result.itl_s if result.itl_s is not None else '-'}  "
                f"out_tok={result.tokens_reported if result.tokens_reported is not None else '-'}  "
                f"tools={tools}  "
                f"text={(result.output_text[:80] + '…') if len(result.output_text) > 80 else result.output_text}"
            )
            if result.error and not result.ok:
                print(f"       error={result.error[:200]}")
    suite_wall_s = time.perf_counter() - suite_started

    summary = _summarize(results)
    summary["suite_wall_time_s"] = round(suite_wall_s, 4)
    mcp_meta = {
        "server_label": args.mcp_label,
        "server_url": args.mcp_url,
        "allowed_tools": allowed,
    } if enable_mcp else None
    rag_meta = {
        "mode": rag_mode,
        "vector_store_ids": vector_store_ids if rag_mode == "file_search" else None,
        "gateway_mcp_url": args.mcp_url if rag_mode == "gateway_mcp" else None,
        "retrieve_api": (
            "bedrock-agent-runtime:Retrieve" if rag_mode == "bedrock_kb" else None
        ),
        "knowledge_base_id": (
            resolve_knowledge_base_api_id()
            if rag_mode in {"gateway_mcp", "bedrock_kb"}
            else None
        ),
        "knowledge_base_name": (
            resolve_knowledge_base_name()
            if rag_mode in {"gateway_mcp", "bedrock_kb"}
            else None
        ),
        "kb_embedding_model": (
            resolve_kb_embedding_model()
            if rag_mode in {"gateway_mcp", "bedrock_kb"}
            else None
        ),
        "mcp_tools": (
            resolve_kb_gateway_mcp_tool_names() if rag_mode == "gateway_mcp" else None
        ),
        "enable_rag": True,
    } if enable_rag else None
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": stamp,
        "scenario": scenario,
        "target": args.target,
        "base_url": base_url,
        "model": args.model,
        "prompt": prompt,
        "instructions": instructions,
        "prompt_files": prompt_meta,
        "mcp": mcp_meta,
        "rag": rag_meta,
        "summary": summary,
        "trials": [_trial_dict(r) for r in sorted(results, key=lambda r: r.trial_id or 0)],
    }
    if args.target.startswith("bedrock"):
        report["bedrock"] = {
            "gateway_url": gateway_url if args.target == "bedrock-gateway" else None,
            "inference_url": (
                resolve_gateway_urls(gateway_url)["responses_url"]
                if args.target == "bedrock-gateway" and gateway_url
                else None
            ),
            "runtime_arn": (
                args.bedrock_runtime_arn if args.target == "bedrock-runtime" else None
            ),
            "runtime_mcp_url": args.mcp_url if args.target == "bedrock-runtime" else None,
            "runtime_qualifier": args.bedrock_runtime_qualifier or None,
            "region": args.bedrock_region or None,
            "auth_mode": args.bedrock_auth,
            "knowledge_base_id": resolve_knowledge_base_id() or None,
            "kb_embedding_model": resolve_kb_embedding_model() or None,
        }
    if enable_mcp:
        report["jira_responses_file"] = str(jira_path)
    if enable_rag:
        report["rag_responses_file"] = str(rag_path)

    if enable_mcp:
        write_jira_responses_file(
            jira_path,
            stamp=stamp,
            prompt=prompt,
            model=args.model,
            mcp=mcp_meta or {},
            results=results,
        )
    if enable_rag:
        write_rag_responses_file(
            rag_path,
            stamp=stamp,
            prompt=prompt,
            model=args.model,
            rag=rag_meta or {},
            results=results,
        )

    if use_phoenix:
        from benchmarks.phoenix_semantic import (
            PhoenixEvalRow,
            log_evals_to_phoenix,
            run_phoenix_evals,
            scenario_eval_config,
        )

        eval_cfg = scenario_eval_config(scenario)
        print("\nRunning Phoenix semantic + trajectory judges…")
        judge_started = time.perf_counter()
        eval_rows = [
            PhoenixEvalRow(
                trial_id=r.trial_id if r.trial_id is not None else i,
                input=prompt,
                output=r.output_text,
                trajectory=r.trajectory,
                tools=tools_desc,
                reference_trajectory=eval_cfg["reference_trajectory"],
                orchestration_time_s=r.latency_s,
                mcp_list_tools=r.mcp_list_tools,
                mcp_calls=r.mcp_calls,
                tool_events=r.tool_events,
                http_ok=r.ok,
                response_id=r.response_id,
            )
            for i, r in enumerate(sorted(results, key=lambda x: x.trial_id or 0))
        ]
        phoenix_summary = run_phoenix_evals(
            eval_rows,
            require_trajectory=not args.skip_trajectory_gate,
            agent_model=args.model,
            judge_model=args.judge_model or None,
            scenario=scenario,
        )
        judge_wall_s = time.perf_counter() - judge_started
        phoenix_summary["judge_wall_time_s"] = round(judge_wall_s, 4)
        # Attach per-request total times onto phoenix trial rows too.
        for t in phoenix_summary.get("trials") or []:
            tid = t.get("trial_id")
            match = next((r for r in results if r.trial_id == tid), None)
            if match is not None:
                t["total_time_s"] = round(match.total_time_s, 4)
        report["phoenix"] = phoenix_summary
        log_evals_to_phoenix(phoenix_summary, args.phoenix_project)
        summary["semantic_success_rate"] = phoenix_summary["semantic_success_rate"]
        summary["trajectory_success_rate"] = phoenix_summary["trajectory_success_rate"]
        summary["combined_success_rate"] = phoenix_summary["combined_success_rate"]
        summary["judge_model"] = phoenix_summary.get("judge_model")
        summary["judge_wall_time_s"] = round(judge_wall_s, 4)
        report["summary"] = summary

    out_path.write_text(json.dumps(report, indent=2))
    _print_final_report(summary, suite_wall_s)
    print(f"Wrote {out_path}")
    if enable_mcp:
        print(f"Wrote Jira responses {jira_path}")
        for r in sorted(results, key=lambda x: x.trial_id or 0):
            keys = [i.get("key") for i in r.jira_issues if i.get("key")]
            if keys:
                print(f"  req={r.trial_id} issues: {', '.join(keys)}")
            elif r.mcp_tool_responses:
                names = [c.get("name") for c in r.mcp_tool_responses]
                print(
                    f"  req={r.trial_id} mcp calls: {names} "
                    "(see jira-responses file for payloads)"
                )
    if enable_rag:
        print(f"Wrote RAG responses {rag_path}")
        for r in sorted(results, key=lambda x: x.trial_id or 0):
                print(
                    f"  req={r.trial_id} rag_time_s={r.rag_time_s} "
                    f"mcp_time_s={r.mcp_time_s} "
                    f"time_to_rag_s={r.time_to_rag_s} rag_calls={r.rag_calls} "
                    f"mcp_calls={r.mcp_calls}"
                )

    if use_phoenix:
        rate = summary.get("combined_success_rate", 0.0)
        return 0 if rate >= 1.0 else 1
    return 0 if summary["ok"] == summary["requests"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
