# Jira create-issue prompts

| File | Role |
|------|------|
| `system.txt` | System / instructions prompt (checked in) |
| `user.txt.example` | Example user prompt with placeholders |
| `user.txt` | **Edit this** before each run (local; gitignored) |

## Run

```bash
# 1. Edit the user prompt
$EDITOR benchmarks/prompts/jira_create/user.txt

# 2. Execute (uses system.txt + user.txt)
./scripts/benchmark_mcp_jira_create.sh --requests 1
```

Optional overrides:

```bash
./scripts/benchmark_mcp_jira_create.sh \
  --system-prompt-file benchmarks/prompts/jira_create/system.txt \
  --user-prompt-file benchmarks/prompts/jira_create/user.txt \
  --requests 1
```
