# Evaluation record

Generated: 2026-10-22T22:59:22.865579+00:00. Task revision SHA-256: `37426b6d562d2ef98657642a5f98420bb8ea4e4840a3da46379ad3856995b04b`.

The task remains the frozen 333-test revision. Both images previously completed public-source cold builds. Accepted validation: Oracle 333/333, reward 1; independent alternative 333/333, reward 1; Nop 328/333, reward 0, with five performance failures. All 28 current upstream static checks passed again after verified author metadata was filled in.

## Actual local model attempts

| Trial | Configuration | Reward | Status / validity |
| --- | --- | ---: | --- |
| deepseek-standard-1 | claude-code / deepseek-flash / max | 1 | completed; valid legitimate model completion |
| codex-standard-1 | codex / openai/gpt-6.1-sol / xhigh | 0 | completed; invalid infrastructure attempt |
| codex-adversarial-1 | codex / openai/gpt-6.1-sol / xhigh | pending | execution-error; unavailable: provider safety restriction; not a valid task failure |
| codex-standard-2 | codex / openai/gpt-6.1-sol / xhigh | 1 | completed; valid legitimate graded model completion |
| deepseek-adversarial-1 | claude-code / deepseek-flash / max | pending | execution-error; withheld: do not route the same blocked adversarial prompt through another provider |
| deepseek-standard-2 | claude-code / deepseek-flash / max | 1 | completed; valid legitimate graded model completion |
| codex-standard-3 | codex / openai/gpt-6.1-sol / xhigh | 1 | completed; valid graded model execution; confirmed ungraded compatibility defect |
| deepseek-standard-3 | claude-code / deepseek-flash / max | 1 | completed; valid legitimate graded model completion |
| codex-standard-1-replacement | codex / openai/gpt-6.1-sol / xhigh | 1 | completed; valid legitimate graded model completion |

Standard configurations still unfinished or not started at this timestamp: none. A missing or unfinished result is not a model failure. The original `codex-standard-1` attempt is excluded: the CLI preload omitted a required helper, and its trajectory shows that all workspace commands failed. Its observed reward is preserved; one independent replacement is scheduled after the complete native package repair.

Observed standard results: 6/6 completed required standard trials received reward 1; 0 required standard runs received reward 0. Execution validity and semantic limitations are recorded per row. The observed pass rate alone does not establish professional difficulty.

The Codex adversarial request was rejected by the provider safety check before task actions. It was stopped and will not be retried or rephrased. The matching DeepSeek adversarial request was withheld rather than routing the same blocked prompt through another provider. Neither is a valid model failure or a successful anti-cheating validation.

## Authentication and runtime

Codex `login status` reported ChatGPT authentication. Its successful Harbor smoke session recorded `gpt-6.1-sol` and `xhigh`. DeepSeek's successful Claude Code smoke emitted a real request to the official Anthropic endpoint with `deepseek-flash`, `output_config.effort=max`, and HTTP 200. Read-only native CLI artifacts are pinned to Codex 0.163.0 and Claude Code 2.1.308. The complete Codex package includes its code-mode helper and runtime resources.

A metadata-only local DeepSeek adapter holds the protected key outside the agent container and forwards unchanged inference bodies to the official TLS endpoint. It logs model, effort, status and timing; it does not log messages, headers or credentials. Token-count requests have no reasoning-effort field. Provider cost estimates in agent logs are not billing evidence.

Harbor removes agent CLI mounts and Compose overlays before starting the separate verifier. The canonical one-CPU, 2-GiB budgets, 14,400-second agent timeout, 1,200-second verifier timeout, numerical policies and performance limits are unchanged. Trial concurrency is at most two jobs, one per provider; raw commands, native exit codes, trajectories and reports are retained locally. The reserved-hour cutoff is 2026-10-23 06:13:08 UTC (14:13 China time).

## Review and remaining concerns

The source-based rubric self-review is manual, not an official verdict. The single substituted automated review uses the pinned upstream rubric, instruction and word-count helper. Its actual status is: execution finished; inspect verdict artifact; verdict counts {"fail": 2, "not_applicable": 1, "pass": 33, "warn": 0}. It reviews the task package, not the complete model-experiment record or post-hoc diagnostic. A reward for that review means a structurally valid verdict artifact, not that all task criteria passed. No official upstream hosted review or PR workflow result is claimed.

The instruction and four README explanations were AI-generated and are disclosed as such. They do not satisfy the upstream human-authorship requirement. The GitHub display identity comes from existing Git configuration and a matching email on the connected account; no legal-name inference or survival-analysis expertise claim is made.

An ungraded diagnostic found a concrete compatibility gap in `codex-standard-3`, despite its observed 333/333 reward 1. For two records with event indicators `[True, False]`, times `[1, 2]`, equal float16 risks and `tied_tol=-1e-100`, pristine 0.28.0 returns `(0.5, 0, 0, 1, 0)` while that submission returns `(0.0, 0, 1, 0, 0)`. NumPy casts the tolerance to float16 negative zero. The standalone diagnostic exited 0; other already-reviewed submissions matched pristine behavior. See [the diagnostic record](posthoc-numerical-check.json). This is a grading false positive for the broader compatibility contract, not a new graded failure. The frozen verifier, task requirements and all reported rewards remain unchanged.

GitHub publication was initially blocked by an invalid saved credential (401). The connected app can write repository files but cannot create a repository; browser interaction was stopped by its URL-policy check. No browser retry or safety-control bypass was attempted. A complete sanitized local repository and ZIP are prepared; a remote URL is verified only after an actual successful publication.

Task difficulty and model-success claims require the completed trajectory analysis. Do not interpret preliminary rewards, authentication smokes, or the blocked adversarial requests as established frontier-model difficulty or verifier robustness.

Readiness: this is an executable experimental artifact, not a fully compliant submission. Human authorship, unavailable adversarial checks, the demonstrated ungraded compatibility gap, and the missing verified GitHub publication remain explicit limitations. No task redesign or threshold change was made to improve failure statistics.

## Final review follow-up

The substituted review returned 33 pass, 2 fail and 1 not applicable (native Harbor exit 0). Its JSON-format reward is not an approval of the task. The generated-authorship failure is confirmed. Its schema failure follows an outdated field inventory: the pinned Harbor parser recognizes `network_mode` and parses `public`; its source deprecates `allow_internet`. The raw failure is retained, with [independent parser evidence](network-field-schema-check.json) and [the technical follow-up](review-followup.json). The frozen configuration is unchanged.

All six required standard runs completed with 333/333 and observed reward 1. Full trajectory and source reviews found general input-dependent implementations, including a vectorized sorted-block alternative. The known ungraded Codex compatibility mismatch remains a real grading limitation. Neither unavailable adversarial result is counted as a model failure.

## Publication portability closeout

The final archive audit detected line-ending conversion in two earlier Git-generated ZIPs. Repository attributes now preserve exact bytes. A fresh Linux checkout with `core.autocrlf=true` matched all 26 frozen task-file hashes. Both Docker images built from that checkout (native exits 0), and all 28 upstream static scripts passed again. These builds used the normal build cache; prior public-source cold-build results remain separate.

Fresh checkout controls: Oracle 333/333, reward 1; Nop 327/333, reward 0, with 6 failures. Both Harbor commands exited 0 without a trial exception and produced complete CTRF reports. No model trials were repeated and no task file, statistical rule, performance limit or resource setting changed. See [the checkout evidence](packaging-portability.json) and [the preserved packaging failure](line-ending-packaging-defect.json). The earlier `standard-matrix` and `reviewed` ZIPs are superseded because their task bytes did not match the frozen manifest.
