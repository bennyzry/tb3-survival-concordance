# Genuine v0.3.0 model evaluation

Report generated: 2026-10-23T05:37:43.536575+00:00. Evaluation target: exact clean checkout of commit `69fd982b09b69d8223bd8626fa8f227a52760576`, tagged [`verifier-v0.3.0`](https://github.com/bennyzry/tb3-survival-concordance/tree/verifier-v0.3.0). Task revision SHA-256: `df90c60c5d428a34fcc24fbb493655d38bee8d56ea6505ab62c511db3f7b2c4e`; 365 graded tests.

These are new independent Harbor model executions on the final task. Each started from the supplied unchanged 0.28.0 implementation in a fresh container. No saved model submission, Oracle implementation, hidden verifier, or earlier trajectory was supplied to the agent. The task's 26 files were checked against the published manifest before launch and after evidence collection. The instruction, verifier, Oracle, timeouts, numeric policies, performance thresholds, resource allocation and agent-facing environment are unchanged.

## New final-version trials

| Genuine final-version trial | Model / effort | Passed | Reward | Validity |
| --- | --- | ---: | ---: | --- |
| `v030-codex-standard-1` | openai/gpt-6.1-sol / xhigh | 365/365 | 1 | valid genuine final-version graded model completion |
| `v030-codex-standard-2` | openai/gpt-6.1-sol / xhigh | 365/365 | 1 | valid genuine final-version graded model completion |
| `v030-codex-standard-3` | openai/gpt-6.1-sol / xhigh | 365/365 | 1 | valid genuine final-version graded model completion |
| `v030-deepseek-standard-1` | deepseek-flash / max | 365/365 | 1 | valid genuine final-version graded model completion |
| `v030-deepseek-standard-2` | deepseek-flash / max | 365/365 | 1 | valid genuine final-version graded model completion |
| `v030-deepseek-standard-3` | deepseek-flash / max | 365/365 | 1 | valid genuine final-version graded model completion |

At this report timestamp, 6 executions have structurally complete native results, binary rewards and complete 365-case CTRF reports; 6 also have complete trajectory/source reviews. Reviewed outcomes: 6 reward-1 successes and 0 reward-0 failures. Unfinished launched IDs: none. A missing report, provider error, cutoff interruption or verifier infrastructure failure is not a valid model failure. Observed successes are retained.

All six final-version standard attempts succeeded. This does not meet the intended model-failure-rate target and does not demonstrate frontier-level difficulty.

The [machine-readable summary](summary.json) records native exits, task checksums, model configuration, source hashes, agent/verifier timing and validity. Each completed graded trial directory contains the full sanitized ATIF trajectory, effective Harbor configuration, native result, submitted source and hash manifest, complete CTRF results, protected-verifier environment/resource records, launcher output, and its trajectory analysis when available. Exact private originals remain preserved locally; public text exports redact credentials/private host paths. Native agent session/configuration files that may contain authentication material are excluded.

## Runtime and configuration evidence

The persistent WSL supervisor began the first independent trial for each provider at 11:50:36 CST on October 23, initially running one attempt per provider at a time. After all three Codex attempts finished, DeepSeek trial 3 started at 12:54:03 CST in the freed slot while trial 2 completed local tests. Total model-trial concurrency remained at most two, with each task keeping its one-CPU/2-GiB allocation. This is below the [published DeepSeek Flash concurrency limit](https://api-docs.deepseek.com/quick_start/rate_limit/); actual per-request status is retained. The same immutable trial ID and exclusive log-file claim prevent the original sequential coordinator from launching a duplicate. The [handoff record](orchestration-change.json) and controller diagnostics preserve this scheduling change separately from model outcomes.

The controllers stop launching by 13:15 CST and interrupt any remaining execution at the 14:00 evidence cutoff; such interruptions are not model failures. These controller deadlines do not change the frozen 14,400-second task agent timeout or 1,200-second verifier timeout.

The existing Harbor `0.23.1.dev202609170426` and integrity-checked native Codex `0.163.0` / Claude Code `2.1.308` were reused without reinstalling tools. Codex uses existing ChatGPT subscription authentication. Native Codex turn-context fields record the effective model/effort when available. DeepSeek uses the protected prepaid account through the existing metadata-only relay to the official Anthropic-compatible endpoint; per-request metadata records the actual `deepseek-flash` model and `output_config.effort=max`. An alias is not independent attestation of the backend snapshot. Cost fields in agent logs are estimates, not invoices; no credits were purchased or billing settings changed.

Reproduce one independent standard attempt with the existing public launcher and fresh ID:

```bash
git checkout --detach 69fd982b09b69d8223bd8626fa8f227a52760576
# Prepare the pinned official CLI artifacts as described in the root README.
# Point CODEX_AUTH_JSON_PATH to your existing private ChatGPT auth.json.
python3 scripts/run_standard.py codex --id independent-v030-codex
# DeepSeek reads the protected user-owned key file only at runtime.
python3 scripts/run_standard.py deepseek --id independent-v030-deepseek
```

## Trajectory and source findings

- **v030-codex-standard-1:** Uses a compiled C rank-counting helper with a Fenwick tree, NumPy searches using the actual dtype subtraction for inclusive tie boundaries, and ordered weighted accumulation. It retains the original loop for small or irregular inputs and includes both the helper source and the exact shared library under the declared artifact scope. The isolated final-version verifier passed all 365 cases, including the 32 scalar-tolerance and warning regressions and the unchanged scale workloads. This is a genuine reward-1 success, not an offline regrade or an infrastructure failure.
- **v030-deepseek-standard-1:** Uses a vectorized wavelet-matrix prefix counter over time-ordered risk ranks, NumPy subtraction-based binary searches for nontransitive tie intervals, and separate same-time censor contributions. It retains the release loop for dtypes whose arithmetic cannot support ordered queries and preserves weighted accumulation and warning categories in the single submitted metrics.py module. The isolated final-version verifier passed all 365 cases. This independent wavelet-matrix implementation is a genuine reward-1 success and demonstrates acceptance of an algorithm different from the Oracle and the first Codex submission.
- **v030-codex-standard-2:** Uses a Python Fenwick tree over risk ranks, reverse time groups, dtype-preserving binary searches for tie intervals and ordered scalar weighted accumulation. It retains the pairwise implementation for irregular inputs and integer ranges whose subtraction can wrap; a source package overlay and local validation scripts are included in the transferred artifact. All 365 final-version cases passed, with no timeout or trial exception. This is a genuine reward-1 success from a fresh independent Codex session; the implementation uses no compiled helper.
- **v030-deepseek-standard-2:** Uses Python Fenwick rank counts over reverse survival-time groups, vectorized subtraction-based tie-window searches and separate same-time censor contributions. It selects the optimized path by estimated pairwise work and input arithmetic properties, retains the release implementation as a fallback, and includes a package bootstrap for relocated imports. The isolated verifier passed all 365 cases with no timeout or trial exception. The approximately fifty-minute run included lengthy agent-side reference calculations on 200,000-record cohorts; this was ordinary local validation work, not an authentication or infrastructure failure.
- **v030-codex-standard-3:** Uses a C++ rank tree that removes same-time events before querying the remaining subjects, with subtraction-based inclusive tie bounds and 64-bit pair counts. Python retains validation, IPCW weights and ordered NumPy accumulation; non-float64 risks and unusual tolerances use the original loop. The C++ source, exact shared library and Python adapter are all captured under the declared artifact scope. The isolated verifier passed all 365 cases, with no timeout or trial exception. This new v030-codex-standard-3 is a reward-1 success and is distinct from the earlier v0.2.0 codex-standard-3 submission whose separate offline regrade failed.
- **v030-deepseek-standard-3:** Uses Python Fenwick rank counts over reverse survival-time groups, vectorized tie-window searches that evaluate the release's dtype subtraction/comparison, and sorted same-time censor ranks. Ordered weighted accumulation and the original loop for unsupported arithmetic remain in the single submitted metrics.py file. The isolated verifier passed all 365 final-version cases. This is a genuine reward-1 success from a new independent session. Its long local campaigns and intermediate implementation edits are not final benchmark failures.

## Historical results remain separate

| Evidence type | Target and execution | Observed results |
| --- | --- | --- |
| Original six valid standard model trials | v0.2.0, 333 tests, actual agent inference | All six reward 1, 333/333. The first invalid Codex CLI-bootstrap attempt is separately preserved and excluded. |
| Six saved-artifact offline regrades | v0.3.0, 365 tests, no new model inference | Five reward 1; saved `codex-standard-3` reward 0, 357/365. |
| New final-version trials | Exact v0.3.0 commit above, fresh independent agent inference | Only the new table in this report counts toward final-version model trials. |

The [original model record](../evaluation.md) and [verifier revision / offline regrades](../verifier-revision/report.md) remain unchanged. A regrade is diagnostic evidence, not a new original trial. New report-only Git commits do not change the immutable evaluation target.

## Validation and remaining limitations

Preserved v0.3.0 controls: Oracle 365/365, reward 1; independent alternative 365/365, reward 1; unchanged Nop 359/365, reward 0. All 28 upstream static scripts passed, both v0.3.0 images built, and prior public-source cold builds remain recorded separately. No Docker/static/control checks were repeated merely to add model reports. The verifier uses separate root-owned grading and UID/GID 10001 candidate execution under one CPU and 2 GiB RAM, with complete CTRF reporting and binary rewards.

The prior Codex adversarial prompt was rejected by the provider before task actions; the corresponding DeepSeek prompt was withheld. It was not retried, rephrased or rerouted, and adversarial results remain unavailable. The task instruction and four README explanation paragraphs are AI-generated and do not satisfy the upstream human-authorship requirement. The [current evidence review](manual-evidence-review.json) is a manual Codex source/evidence audit with artifact-consistency checks. No official upstream hosted implementation review or PR CI verdict is claimed; preserved substituted and manual reviews retain their limitations. The old schema flag does not justify changing valid `network_mode="public"` configuration. Passing controls or one numerical failure does not establish novelty, full API correctness, frontier-level difficulty, or the employer's requested failure-rate target.
