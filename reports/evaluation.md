# Evaluation record

Generated: 2026-10-22T21:59:35.806472+00:00. Task revision SHA-256: `37426b6d562d2ef98657642a5f98420bb8ea4e4840a3da46379ad3856995b04b`.

The task remains the frozen 333-test revision. Both images previously completed public-source cold builds. Accepted validation: Oracle 333/333, reward 1; independent alternative 333/333, reward 1; Nop 328/333, reward 0, with five performance failures. All 28 current upstream static checks passed again after verified author metadata was filled in.

## Actual local model attempts

| Trial | Configuration | Reward | Status / validity |
| --- | --- | ---: | --- |
| deepseek-standard-1 | claude-code / deepseek-flash / max | 1 | completed; valid-execution; trajectory review pending |
| codex-standard-1 | codex / openai/gpt-6.1-sol / xhigh | 0 | completed; invalid infrastructure attempt |
| codex-adversarial-1 | codex / openai/gpt-6.1-sol / xhigh | pending | execution-error; unavailable: provider safety restriction; not a valid task failure |
| codex-standard-2 | codex / openai/gpt-6.1-sol / xhigh | pending | running; pending |
| deepseek-adversarial-1 | claude-code / deepseek-flash / max | pending | running; withheld: do not route the same blocked adversarial prompt through another provider |

Standard configurations still unfinished or not started at this timestamp: codex-standard-1-replacement, codex-standard-2, codex-standard-3, deepseek-standard-2, deepseek-standard-3. A missing or unfinished result is not a model failure. The original `codex-standard-1` attempt is excluded: the CLI preload omitted a required helper, and its trajectory shows that all workspace commands failed. Its observed reward is preserved; one independent replacement is scheduled after the complete native package repair.

The Codex adversarial request was rejected by the provider safety check before task actions. It was stopped and will not be retried or rephrased. The matching DeepSeek adversarial request was withheld rather than routing the same blocked prompt through another provider. Neither is a valid model failure or a successful anti-cheating validation.

## Authentication and runtime

Codex `login status` reported ChatGPT authentication. Its successful Harbor smoke session recorded `gpt-6.1-sol` and `xhigh`. DeepSeek's successful Claude Code smoke emitted a real request to the official Anthropic endpoint with `deepseek-flash`, `output_config.effort=max`, and HTTP 200. Read-only native CLI artifacts are pinned to Codex 0.163.0 and Claude Code 2.1.308. The complete Codex package includes its code-mode helper and runtime resources.

A metadata-only local DeepSeek adapter holds the protected key outside the agent container and forwards unchanged inference bodies to the official TLS endpoint. It logs model, effort, status and timing; it does not log messages, headers or credentials. Token-count requests have no reasoning-effort field. Provider cost estimates in agent logs are not billing evidence.

Harbor removes agent CLI mounts and Compose overlays before starting the separate verifier. The canonical one-CPU, 2-GiB budgets, 14,400-second agent timeout, 1,200-second verifier timeout, numerical policies and performance limits are unchanged. Trial concurrency is at most two jobs, one per provider; raw commands, native exit codes, trajectories and reports are retained locally. The reserved-hour cutoff is 2026-10-23 06:13:08 UTC (14:13 China time).

## Review and remaining concerns

The source-based rubric self-review is manual, not an official verdict. A single substituted automated review using the pinned upstream rubric, instruction and word-count helper is queued after the standard trials and the infrastructure replacement; its actual status is recorded separately. No official upstream hosted review or PR workflow result is claimed.

The instruction and four README explanations were AI-generated and are disclosed as such. They do not satisfy the upstream human-authorship requirement. The GitHub display identity comes from existing Git configuration and a matching email on the connected account; no legal-name inference or survival-analysis expertise claim is made.

GitHub publication was initially blocked by an invalid saved credential (401). The connected app can write repository files but cannot create a repository; browser interaction was stopped by its URL-policy check. No browser retry or safety-control bypass was attempted. A complete sanitized local repository and ZIP are prepared; a remote URL is verified only after an actual successful publication.

Task difficulty and model-success claims require the completed trajectory analysis. Do not interpret preliminary rewards, authentication smokes, or the blocked adversarial requests as established frontier-model difficulty or verifier robustness.
