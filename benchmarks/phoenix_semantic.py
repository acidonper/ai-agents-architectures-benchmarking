"""Phoenix semantic + trajectory evaluation for agentic MCP benchmarks.

Metrics produced:
  - orchestration_time_s  — wall-clock E2E agent turn (LLM + MCP)
  - semantic_success      — LLM-as-judge: answer fulfills the user task
  - trajectory_correct    — LLM-as-judge: tool path is sensible
  - success_rate          — aggregate of semantic (+ optional trajectory) passes
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

SEMANTIC_SUCCESS_PROMPT_LIST = """
You evaluate whether an agent completed this Jira task successfully.

[User question]
{input}

[Agent trajectory / tool steps]
{trajectory}

[Agent final answer]
{output}

Mark CORRECT if ALL of these hold:
1. The final answer lists Jira issues for project SUP with key/status/summary style details,
   OR clearly states that tools returned zero issues / an access error.
2. The answer is consistent with the tool trajectory (does not invent a large set of keys
   when tools clearly failed or returned nothing).
3. The response is usable for a human who asked to list issues in SUP.

Mark INCORRECT only if the answer invents issues without tool support, ignores the task,
or is empty/nonsensical.

Classify as correct or incorrect.
"""

SEMANTIC_SUCCESS_PROMPT_CREATE = """
You evaluate whether an agent successfully created a Jira issue via MCP tools.

[User question]
{input}

[Agent trajectory / tool steps]
{trajectory}

[Agent final answer]
{output}

Mark CORRECT if ALL of these hold:
1. The trajectory includes a successful jira_create_issue (or equivalent create) MCP call,
   OR the answer clearly reports a real tool/API failure.
2. The final answer reports a created issue key from tool output (e.g. SUP-123),
   matching the create request fields when available.
3. The agent does not claim success without a create tool call.

Mark INCORRECT if it invents a key, skips create tools, or ignores the user request.

Classify as correct or incorrect.
"""

SEMANTIC_SUCCESS_PROMPT_RAG = """
You evaluate whether an agent successfully used RAG and MCP together.

[User question]
{input}

[Agent trajectory / tool steps]
{trajectory}

[Agent final answer]
{output}

Mark CORRECT if ALL of these hold:
1. The trajectory includes file_search / retrieval against a vector store to load
   procedure or knowledge context when the task depends on company docs.
2. When the user asks for a Jira action, the trajectory includes the appropriate
   MCP tool call (e.g. jira_create_issue), OR clearly reports a real tool/API failure.
3. The final answer is consistent with retrieved context and tool outputs
   (does not invent issue keys or skip required procedure fields).

Mark INCORRECT if it skips retrieval when docs are needed, skips MCP when an
action was requested, invents results, or ignores the user request.

Classify as correct or incorrect.
"""

TRAJECTORY_PROMPT_LIST = """
You evaluate the agent's tool-calling trajectory for a Jira MCP task.

[User question]
{input}

[Available tools context]
{tools}

[Actual trajectory]
{trajectory}

[Final answer]
{output}

[Expected reference trajectory]
{reference_trajectory}

Mark CORRECT if:
- mcp_list_tools appears (or tools were already known), AND
- a project listing/search tool is called with SUP
  (jira_get_project_issues with project_key=SUP OR jira_search with JQL containing SUP), AND
- the agent does not primarily rely on inventing random issue keys via jira_get_issue before searching.

A short efficient path of 2 steps (list tools → get_project_issues/search) is CORRECT.

Classify as correct or incorrect.
"""

TRAJECTORY_PROMPT_CREATE = """
You evaluate the agent's tool-calling trajectory for creating a Jira issue.

[User question]
{input}

[Available tools context]
{tools}

[Actual trajectory]
{trajectory}

[Final answer]
{output}

[Expected reference trajectory]
{reference_trajectory}

Mark CORRECT if:
- mcp_list_tools appears (or tools were already known), AND
- jira_create_issue (or equivalent create tool) is called with fields from the user request
  (project key, summary, etc.), AND
- the path does not invent issue keys before create returns.

Classify as correct or incorrect.
"""

TRAJECTORY_PROMPT_RAG = """
You evaluate the agent's tool-calling trajectory for a combined RAG + MCP task.

[User question]
{input}

[Available tools context]
{tools}

[Actual trajectory]
{trajectory}

[Final answer]
{output}

[Expected reference trajectory]
{reference_trajectory}

Mark CORRECT if:
- file_search is invoked against the configured vector store(s) to retrieve
  procedure / knowledge context, AND
- MCP tools are used for the requested Jira action (list/create/update) when applicable, AND
- the path does not invent issue keys before tools return them.

A short path of file_search → mcp_call → final answer is CORRECT.

Classify as correct or incorrect.
"""

# Back-compat aliases used by older call sites
SEMANTIC_SUCCESS_PROMPT = SEMANTIC_SUCCESS_PROMPT_LIST
TRAJECTORY_PROMPT = TRAJECTORY_PROMPT_LIST


def code_trajectory_ok_list(trajectory: str, mcp_calls: int) -> bool:
    """Deterministic trajectory gate for the SUP list-issues task."""
    t = (trajectory or "").lower()
    if mcp_calls <= 0 and "mcp_call" not in t:
        return False
    has_project_tool = (
        "jira_get_project_issues" in t
        or ("jira_search" in t and "sup" in t)
    )
    invented_first = False
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in lines:
        if "mcp_call" not in ln:
            continue
        if "jira_get_issue" in ln and "error" in ln and "sup-" in ln:
            invented_first = True
        if "jira_get_project_issues" in ln or "jira_search" in ln:
            break
    return has_project_tool and not (
        invented_first and "jira_get_project_issues" not in t and "jira_search" not in t
    )


def code_trajectory_ok_create(trajectory: str, mcp_calls: int) -> bool:
    """Deterministic trajectory gate for create-issue."""
    t = (trajectory or "").lower()
    if mcp_calls <= 0 and "mcp_call" not in t:
        return False
    return "jira_create_issue" in t or "create_issue" in t


def code_trajectory_ok_rag(trajectory: str, mcp_calls: int = 0) -> bool:
    """Deterministic trajectory gate for combined RAG + MCP."""
    t = (trajectory or "").lower()
    has_rag = "file_search" in t
    has_mcp = mcp_calls > 0 or "mcp_call" in t
    return has_rag and has_mcp


def code_trajectory_ok(trajectory: str, mcp_calls: int, scenario: str = "list") -> bool:
    if scenario == "create":
        return code_trajectory_ok_create(trajectory, mcp_calls)
    if scenario == "rag":
        return code_trajectory_ok_rag(trajectory, mcp_calls)
    return code_trajectory_ok_list(trajectory, mcp_calls)


DEFAULT_REFERENCE_TRAJECTORY_LIST = (
    "1) mcp_list_tools on jira MCP server\n"
    "2) mcp_call jira_search with JQL `project = SUP` "
    "(or jira_get_project_issues for SUP) — do NOT invent issue keys\n"
    "3) optional follow-up mcp_call only for details on keys returned by step 2\n"
    "4) final message summarizing only tool-returned issues (key, status, summary)"
)

DEFAULT_REFERENCE_TRAJECTORY_CREATE = (
    "1) mcp_list_tools on jira MCP server\n"
    "2) mcp_call jira_create_issue with project/summary/description/type from the user prompt\n"
    "3) final message reporting the created issue key from tool output"
)

DEFAULT_REFERENCE_TRAJECTORY_RAG = (
    "1) file_search against the configured vector store(s) for company procedure/docs\n"
    "2) mcp_list_tools on jira MCP server (if needed)\n"
    "3) mcp_call for the requested Jira action using fields from RAG + user prompt\n"
    "4) final message summarizing retrieval + tool result (e.g. created issue key)"
)

DEFAULT_REFERENCE_TRAJECTORY = DEFAULT_REFERENCE_TRAJECTORY_LIST


def scenario_eval_config(scenario: str) -> dict[str, Any]:
    if scenario == "create":
        return {
            "semantic_prompt": SEMANTIC_SUCCESS_PROMPT_CREATE,
            "trajectory_prompt": TRAJECTORY_PROMPT_CREATE,
            "reference_trajectory": DEFAULT_REFERENCE_TRAJECTORY_CREATE,
        }
    if scenario == "rag":
        return {
            "semantic_prompt": SEMANTIC_SUCCESS_PROMPT_RAG,
            "trajectory_prompt": TRAJECTORY_PROMPT_RAG,
            "reference_trajectory": DEFAULT_REFERENCE_TRAJECTORY_RAG,
        }
    return {
        "semantic_prompt": SEMANTIC_SUCCESS_PROMPT_LIST,
        "trajectory_prompt": TRAJECTORY_PROMPT_LIST,
        "reference_trajectory": DEFAULT_REFERENCE_TRAJECTORY_LIST,
    }


@dataclass
class PhoenixEvalRow:
    trial_id: int
    input: str
    output: str
    trajectory: str
    tools: str
    reference_trajectory: str
    orchestration_time_s: float
    mcp_list_tools: int
    mcp_calls: int
    tool_events: list[str]
    http_ok: bool
    response_id: str | None
    raw_output: Any = None


def configure_phoenix_workdir(root: Path | None = None) -> Path:
    """Keep Phoenix state inside the repo (avoids ~/.phoenix permission issues)."""
    base = root or Path(__file__).resolve().parent
    work = Path(os.environ.get("PHOENIX_WORKING_DIR", base / ".phoenix"))
    work.mkdir(parents=True, exist_ok=True)
    os.environ["PHOENIX_WORKING_DIR"] = str(work)
    return work


def extract_mcp_tool_responses(raw_output: Any) -> list[dict[str, Any]]:
    """Pull mcp_call results (args + Jira tool output) from a Responses payload."""
    calls: list[dict[str, Any]] = []

    def _parse_maybe_json(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
        return value

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("type") == "mcp_call":
                args = _parse_maybe_json(
                    obj.get("arguments") or obj.get("input") or obj.get("params")
                )
                output = _parse_maybe_json(obj.get("output") or obj.get("result"))
                calls.append(
                    {
                        "id": obj.get("id"),
                        "name": obj.get("name") or obj.get("tool_name"),
                        "server_label": obj.get("server_label"),
                        "status": obj.get("status")
                        or ("error" if obj.get("error") else "ok"),
                        "error": obj.get("error"),
                        "arguments": args,
                        "output": output,
                    }
                )
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(raw_output)
    return calls


def extract_jira_issues(mcp_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best-effort flatten of Jira issue dicts from MCP tool outputs."""
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()

    def consider(item: Any) -> None:
        if not isinstance(item, dict):
            return
        key = item.get("key") or item.get("issue_key")
        fields = item.get("fields") if isinstance(item.get("fields"), dict) else {}
        summary = item.get("summary") or fields.get("summary") or item.get("title")
        status = item.get("status")
        if isinstance(status, dict):
            status = status.get("name") or status.get("status")
        elif not status and fields:
            st = fields.get("status")
            status = st.get("name") if isinstance(st, dict) else st
        if not key and not summary:
            return
        key_s = str(key) if key is not None else ""
        if key_s and key_s in seen:
            return
        if key_s:
            seen.add(key_s)
        issues.append(
            {
                "key": key_s or None,
                "status": status,
                "summary": summary,
            }
        )

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6 or node is None:
            return
        if isinstance(node, dict):
            if isinstance(node.get("issues"), list):
                for issue in node["issues"]:
                    consider(issue)
            if isinstance(node.get("values"), list):
                for issue in node["values"]:
                    consider(issue)
            consider(node)
            for key, value in node.items():
                if key in {"issues", "values"}:
                    continue
                walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node:
                consider(item)
                walk(item, depth + 1)

    for call in mcp_calls:
        walk(call.get("output"))
    return issues


def format_trajectory(raw_output: Any, tool_events: list[str] | None = None) -> str:
    """Build a human-readable step list from a Responses API payload."""
    steps: list[str] = []
    n = 0

    def walk(obj: Any) -> None:
        nonlocal n
        if isinstance(obj, dict):
            typ = obj.get("type")
            if typ == "mcp_list_tools":
                n += 1
                tools = obj.get("tools") or []
                names = []
                for t in tools:
                    if isinstance(t, dict):
                        names.append(str(t.get("name") or t.get("id") or t))
                    else:
                        names.append(str(t))
                steps.append(
                    f"{n}. mcp_list_tools  tools=[{', '.join(names[:12])}"
                    f"{'…' if len(names) > 12 else ''}]"
                )
            elif typ == "mcp_call":
                n += 1
                name = obj.get("name") or obj.get("tool_name") or "unknown"
                args = obj.get("arguments") or obj.get("input") or obj.get("params")
                if isinstance(args, (dict, list)):
                    args_s = json.dumps(args)[:400]
                else:
                    args_s = str(args)[:400] if args is not None else ""
                err = obj.get("error")
                status = obj.get("status") or ("error" if err else "ok")
                steps.append(
                    f"{n}. mcp_call name={name} status={status} args={args_s}"
                )
            elif typ in {"function_call", "file_search_call"}:
                n += 1
                steps.append(
                    f"{n}. {typ} name={obj.get('name')} args={str(obj.get('arguments'))[:300]}"
                )
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    if raw_output is not None:
        walk(raw_output)

    if not steps and tool_events:
        for i, ev in enumerate(tool_events, start=1):
            steps.append(f"{i}. {ev}")

    return "\n".join(steps) if steps else "(no tool steps recorded)"


def ensure_phoenix_tracing(project_name: str = "jira-mcp-agentic") -> Any:
    """Register Phoenix OTEL tracer; starts a local UI if no collector is set."""
    configure_phoenix_workdir()
    from phoenix.otel import register

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    kwargs: dict[str, Any] = {
        "project_name": project_name,
        "auto_instrument": True,
    }
    if endpoint:
        kwargs["endpoint"] = endpoint
    else:
        try:
            import phoenix as px

            session = px.launch_app()
            print(f"Phoenix UI: {session.url}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"Phoenix launch_app skipped ({exc}); "
                "set PHOENIX_COLLECTOR_ENDPOINT to attach to a running collector."
            )
    return register(**kwargs)


def trace_agent_span(
    *,
    name: str,
    prompt: str,
    output_text: str,
    trajectory: str,
    orchestration_time_s: float,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Record a root agent span for Phoenix (OpenInference-style attributes)."""
    try:
        from opentelemetry import trace
    except ImportError:
        return

    tracer = trace.get_tracer("benchmarks.agentic")
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("input.value", prompt)
        span.set_attribute("output.value", output_text[:4000])
        span.set_attribute("agent.trajectory", trajectory[:8000])
        span.set_attribute("agent.orchestration_time_s", orchestration_time_s)
        for key, value in (attributes or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, json.dumps(value)[:2000])


def _list_llama_stack_llms(base_url: str, api_key: str | None = None) -> list[str]:
    """Return LLM model ids from Llama Stack /v1/models (skip embeddings)."""
    from urllib.error import URLError
    from urllib.request import Request, urlopen

    url = f"{base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    llms: list[str] = []
    for item in data.get("data") or []:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not mid:
            continue
        meta = item.get("custom_metadata") or item.get("metadata") or {}
        mtype = (meta.get("model_type") or meta.get("type") or "").lower()
        if mtype and mtype != "llm":
            continue
        # Heuristic: skip obvious embedding ids when metadata is missing.
        low = str(mid).lower()
        if "embed" in low or "sentence-transformers" in low:
            continue
        llms.append(str(mid))
    return llms


def resolve_judge_model(
    *,
    agent_model: str | None = None,
    explicit: str | None = None,
) -> tuple[str, str, str]:
    """Resolve Phoenix judge to a Llama Stack OpenAI-compatible endpoint.

    Preference:
      1. PHOENIX_JUDGE_MODEL / explicit
      2. First Stack LLM that differs from the agent model
      3. Agent model / DEFAULT_MODEL (last resort)

    Returns (model_id, base_url_with_v1, api_key).
    """
    stack = (
        os.getenv("PHOENIX_JUDGE_BASE_URL")
        or os.getenv("LLAMA_STACK_BASE_URL")
        or "http://localhost:8321"
    ).rstrip("/")
    # Prefer Llama Stack over OpenAI unless user set a non-OpenAI judge base.
    if "api.openai.com" in stack:
        stack = (
            os.getenv("LLAMA_STACK_BASE_URL") or "http://localhost:8321"
        ).rstrip("/")
        print(
            "Ignoring OpenAI judge URL; Phoenix judge uses Llama Stack. "
            "Set PHOENIX_JUDGE_MODEL to a Stack LLM id."
        )

    base = stack if stack.endswith("/v1") else f"{stack}/v1"
    api_key = (
        os.getenv("PHOENIX_JUDGE_API_KEY")
        or os.getenv("LLAMA_STACK_API_KEY")
        or "EMPTY"
    )

    agent = (
        agent_model
        or os.getenv("BENCH_MODEL")
        or os.getenv("DEFAULT_MODEL")
        or "vllm-inference-1/llama-32-fp8"
    )

    chosen = (
        (explicit or "").strip()
        or os.getenv("PHOENIX_JUDGE_MODEL", "").strip()
        or os.getenv("BENCH_JUDGE_MODEL", "").strip()
    )

    if not chosen:
        llms = _list_llama_stack_llms(base, api_key if api_key != "EMPTY" else None)
        others = [m for m in llms if m != agent]
        if others:
            chosen = others[0]
            print(f"Phoenix judge auto-selected Llama Stack model (≠ agent): {chosen}")
        elif llms:
            chosen = llms[0]
            print(
                f"Phoenix judge using only available Stack LLM: {chosen} "
                "(register another LLM on Llama Stack for a dedicated judge)."
            )
        else:
            chosen = agent
            print(f"Phoenix judge falling back to agent model: {chosen}")

    return chosen, base, api_key


def _judge_llm(agent_model: str | None = None, judge_model: str | None = None):
    """Phoenix evals LLM wrapper — always a Llama Stack OpenAI-compatible model.

    Returns (llm, model_id, base_url).
    """
    from phoenix.evals import LLM

    model_id, base, api_key = resolve_judge_model(
        agent_model=agent_model,
        explicit=judge_model,
    )
    client_kwargs = {"base_url": base, "api_key": api_key}
    print(f"Phoenix judge → {base}  model={model_id}")
    llm = LLM(
        provider="openai",
        model=model_id,
        sync_client_kwargs=client_kwargs,
        async_client_kwargs=client_kwargs,
    )
    return llm, model_id, base


def _parse_score_cell(cell: Any) -> tuple[str, str | None]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return "incorrect", None
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except json.JSONDecodeError:
            return cell.lower(), None
    if isinstance(cell, dict):
        label = str(cell.get("label") or "incorrect").lower()
        expl = cell.get("explanation")
        return label, str(expl) if expl is not None else None
    return "incorrect", None


def _heuristic_labels(
    row: PhoenixEvalRow,
    scenario: str = "list",
) -> tuple[str, str, str, str]:
    """Fallback when the judge model cannot do structured classification."""
    out = (row.output or "").lower()
    traj = (row.trajectory or "").lower()
    has_mcp = row.mcp_calls > 0 or "mcp_call" in traj
    if scenario == "create":
        created = "jira_create_issue" in traj or "create_issue" in traj
        mentions_key = "sup-" in out or "created" in out or "issue key" in out
        semantic = "correct" if row.http_ok and created and (mentions_key or len(out) > 20) else "incorrect"
        trajectory = "correct" if row.http_ok and created else "incorrect"
        return (
            semantic,
            f"heuristic(create): http_ok={row.http_ok} created={created}",
            trajectory,
            f"heuristic(create): requires jira_create_issue; events={row.tool_events}",
        )
    if scenario == "rag":
        searched = "file_search" in traj or any(
            "file_search" in str(e).lower() for e in (row.tool_events or [])
        )
        used_mcp = row.mcp_calls > 0 or "mcp_call" in traj
        empty_ok = any(
            x in out
            for x in ("no relevant", "not found", "no documents", "empty", "could not", "unable", "error")
        )
        semantic = (
            "correct"
            if row.http_ok and searched and used_mcp and (len(out) > 20 or empty_ok)
            else "incorrect"
        )
        trajectory = "correct" if row.http_ok and searched and used_mcp else "incorrect"
        return (
            semantic,
            f"heuristic(rag+mcp): http_ok={row.http_ok} file_search={searched} mcp={used_mcp}",
            trajectory,
            f"heuristic(rag+mcp): requires file_search+mcp_call; events={row.tool_events}",
        )

    mentions_sup = "sup" in out or "sup-" in out or "project sup" in traj
    empty_ok = any(
        x in out
        for x in ("no issues", "0 issues", "none found", "empty", "could not", "unable", "error")
    )
    semantic = "correct" if row.http_ok and has_mcp and (mentions_sup or empty_ok or len(out) > 40) else "incorrect"
    trajectory = "correct" if row.http_ok and has_mcp else "incorrect"
    return (
        semantic,
        f"heuristic: http_ok={row.http_ok} mcp_calls={row.mcp_calls} mentions_sup={mentions_sup}",
        trajectory,
        f"heuristic: requires mcp_call; events={row.tool_events}",
    )


def run_phoenix_evals(
    rows: list[PhoenixEvalRow],
    *,
    require_trajectory: bool = True,
    agent_model: str | None = None,
    judge_model: str | None = None,
    scenario: str = "list",
) -> dict[str, Any]:
    """Run semantic + trajectory ClassificationEvaluators and aggregate rates."""
    configure_phoenix_workdir()
    eval_cfg = scenario_eval_config(scenario)

    if not rows:
        return {
            "semantic_success_rate": 0.0,
            "trajectory_success_rate": 0.0,
            "combined_success_rate": 0.0,
            "orchestration_time_s": {},
            "judge": "none",
            "judge_model": None,
            "judge_base_url": None,
            "scenario": scenario,
            "trials": [],
        }

    for r in rows:
        if scenario == "create" and (
            not r.reference_trajectory
            or r.reference_trajectory == DEFAULT_REFERENCE_TRAJECTORY_LIST
        ):
            r.reference_trajectory = eval_cfg["reference_trajectory"]

    df = pd.DataFrame(
        [
            {
                "trial_id": r.trial_id,
                "input": r.input,
                "output": r.output or "(empty)",
                "trajectory": r.trajectory,
                "tools": r.tools,
                "reference_trajectory": r.reference_trajectory,
                "orchestration_time_s": r.orchestration_time_s,
                "http_ok": r.http_ok,
                "mcp_list_tools": r.mcp_list_tools,
                "mcp_calls": r.mcp_calls,
            }
            for r in rows
        ]
    )

    judge_mode = "phoenix"
    resolved_judge: str | None = None
    judge_base: str | None = None
    semantic_labels: list[str] = []
    trajectory_labels: list[str] = []
    semantic_expl: list[str | None] = []
    trajectory_expl: list[str | None] = []

    try:
        from phoenix.evals import create_classifier, evaluate_dataframe

        llm, resolved_judge, judge_base = _judge_llm(
            agent_model=agent_model,
            judge_model=judge_model,
        )
        semantic_eval = create_classifier(
            name="semantic_success",
            prompt_template=eval_cfg["semantic_prompt"],
            llm=llm,
            choices={"correct": 1.0, "incorrect": 0.0},
        )
        trajectory_eval = create_classifier(
            name="trajectory_correct",
            prompt_template=eval_cfg["trajectory_prompt"],
            llm=llm,
            choices={"correct": 1.0, "incorrect": 0.0},
        )
        scored = evaluate_dataframe(df, [semantic_eval, trajectory_eval])
        for i in range(len(df)):
            s_label, s_e = _parse_score_cell(scored.loc[i, "semantic_success_score"])
            t_label, t_e = _parse_score_cell(scored.loc[i, "trajectory_correct_score"])
            semantic_labels.append(s_label)
            trajectory_labels.append(t_label)
            semantic_expl.append(s_e)
            trajectory_expl.append(t_e)
    except Exception as exc:  # noqa: BLE001
        print(f"Phoenix LLM judge failed ({exc}); falling back to heuristic labels.")
        judge_mode = "heuristic"
        if resolved_judge is None:
            resolved_judge, judge_base, _ = resolve_judge_model(
                agent_model=agent_model,
                explicit=judge_model,
            )
        for r in rows:
            s_l, s_e, t_l, t_e = _heuristic_labels(r, scenario=scenario)
            semantic_labels.append(s_l)
            trajectory_labels.append(t_l)
            semantic_expl.append(s_e)
            trajectory_expl.append(t_e)

    trials: list[dict[str, Any]] = []
    semantic_ok = 0
    trajectory_ok = 0
    combined_ok = 0

    for i, row in enumerate(rows):
        code_ok = code_trajectory_ok(row.trajectory, row.mcp_calls, scenario=scenario)
        t_llm = trajectory_labels[i] == "correct"
        t_ok = t_llm or code_ok
        s_ok = semantic_labels[i] == "correct"
        if not s_ok and code_ok and row.http_ok:
            out = (row.output or "").lower()
            if scenario == "create":
                if any(x in out for x in ("created", "sup-", "issue key", "successfully")):
                    s_ok = True
                    semantic_labels[i] = "correct"
                    semantic_expl[i] = (
                        (semantic_expl[i] or "")
                        + " | overridden_by_code_heuristic: create trajectory ok"
                    )
            elif scenario == "rag":
                if len(out) > 20 and ("sup-" in out or "created" in out or "issue" in out):
                    s_ok = True
                    semantic_labels[i] = "correct"
                    semantic_expl[i] = (
                        (semantic_expl[i] or "")
                        + " | overridden_by_code_heuristic: rag+mcp trajectory ok"
                    )
            elif "sup-" in out or "no issues" in out or "0 issues" in out:
                s_ok = True
                semantic_labels[i] = "correct"
                semantic_expl[i] = (
                    (semantic_expl[i] or "")
                    + " | overridden_by_code_heuristic: SUP issues present with valid trajectory"
                )
        if s_ok:
            semantic_ok += 1
        if t_ok:
            trajectory_ok += 1
            if not t_llm and code_ok:
                trajectory_labels[i] = "correct"
                trajectory_expl[i] = (
                    (trajectory_expl[i] or "")
                    + " | overridden_by_code_trajectory_check"
                )
        passed = s_ok and (t_ok if require_trajectory else True) and row.http_ok
        if passed:
            combined_ok += 1
        trials.append(
            {
                "trial_id": row.trial_id,
                "total_time_s": round(row.orchestration_time_s, 4),
                "orchestration_time_s": round(row.orchestration_time_s, 4),
                "http_ok": row.http_ok,
                "semantic_label": semantic_labels[i],
                "semantic_explanation": semantic_expl[i],
                "trajectory_label": trajectory_labels[i],
                "trajectory_explanation": trajectory_expl[i],
                "trajectory_code_ok": code_ok,
                "passed": passed,
                "trajectory": row.trajectory,
            }
        )

    times = [float(r.orchestration_time_s) for r in rows]
    n = len(rows)
    return {
        "requests": n,
        "scenario": scenario,
        "judge": judge_mode,
        "judge_model": resolved_judge,
        "judge_base_url": judge_base,
        "semantic_success_rate": semantic_ok / n,
        "trajectory_success_rate": trajectory_ok / n,
        "combined_success_rate": combined_ok / n,
        "require_trajectory": require_trajectory,
        "orchestration_time_s": {
            "min": min(times),
            "max": max(times),
            "mean": sum(times) / n,
            "median": sorted(times)[n // 2],
        },
        "trials": trials,
    }


def log_evals_to_phoenix(eval_summary: dict[str, Any], project_name: str = "jira-mcp-agentic") -> None:
    print(
        "Phoenix semantic summary: "
        f"judge={eval_summary.get('judge')}  "
        f"semantic={eval_summary.get('semantic_success_rate'):.0%}  "
        f"trajectory={eval_summary.get('trajectory_success_rate'):.0%}  "
        f"combined={eval_summary.get('combined_success_rate'):.0%}  "
        f"orch_mean={eval_summary.get('orchestration_time_s', {}).get('mean')}s"
    )
    _ = project_name
