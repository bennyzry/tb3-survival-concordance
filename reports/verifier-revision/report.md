# Verifier revision and saved-artifact regrading

Recommendation: use version 0.3.0 as the final experimental artifact. It closes demonstrated API-compatibility gaps without changing the agent's problem, statistical semantics or performance requirements. It does not establish that all employer or upstream acceptance requirements have been met.

Task revision SHA-256: `df90c60c5d428a34fcc24fbb493655d38bee8d56ea6505ab62c511db3f7b2c4e`. The unchanged fallback is [baseline-v0.2.0](https://github.com/bennyzry/tb3-survival-concordance/tree/baseline-v0.2.0), commit `4d2ea842f5c74ae41b817c83e5ab2854c9a767ea`, with its 333-test verifier and original results intact. The actual deadline is **October 23, 2026, 15:00 CST**; the earlier 14:13 cutoff was an internal buffer.

## Why this revision is justified

The published instruction already requires NumPy dtype arithmetic, negative-tolerance compatibility and successful-call warning categories. A 96-case diagnostic crossed subnormal scalar-conversion boundaries in float16, float32 and float64, using both public APIs and equal and mixed risks. Pristine 0.28.0 and the independent pair reference agreed on every case. Five saved implementations matched all 96; `codex-standard-3` differed on 16. A separate subject-replication check at 1,542 records confirmed the mismatch across input sizes and matched the other five implementations and both reference methods.

The verifier adds 28 representative rounding cases, including negative, positive and float64 controls. It also adds four cases for large finite tolerances that overflow when converted to float16 or float32. The latter exposed a real warning-category defect in the original Oracle: its answers were correct, but it suppressed `RuntimeWarning`; all six saved model implementations preserved the warning. The repaired Oracle adds one empty-array comparison before the existing bounded searches to retain the release's scalar-cast warnings. The optimization algorithm is unchanged. A provisional 361-case validation was withdrawn after this warning defect was found; its control results and the intentional orchestrator stop are preserved separately. Final results below use the 365-case revision.

For an analytical example, the six-subject equal-risk cohort has ten comparable pairs and two equal-time event/censor pairs. If the scalar tolerance rounds to negative zero, every risk difference is tied, giving `(0.5, 0, 0, 10, 2)`; a representable negative tolerance gives `(0.0, 0, 10, 0, 2)`. IPCW changes pair weights but preserves these unweighted counts. Replicating every subject `r` times multiplies each pair category by `r**2` and leaves the score unchanged; the replication diagnostic checked that identity independently.

## Executed validation

| Control | Passed | Reward |
| --- | ---: | ---: |
| oracle | 365/365 | 1 |
| alternative | 365/365 | 1 |
| nop | 359/365 | 0 |

Both Docker images built successfully from the ordinary public Dockerfiles with dependency-layer cache enabled; these are not new cold-build claims. The prior successful public-source cold builds remain preserved. All 28 current upstream static scripts passed. Upstream head `e67a2cb7db4d9795c212909b3395d8c22ad60aec` differs from the trusted script snapshot only in a leaderboard record; contribution guidance, rubric and check scripts are unchanged.

All nine control/regrade reports contain 365 unique CTRF test results with consistent totals and binary rewards. Observed cgroups enforce one CPU and 2 GiB; every submitted-code worker ran as UID/GID 10001, with no supplementary groups and `NoNewPrivs=1`. Root-owned grading and report protection are unchanged. Exact commands, native exit codes, image IDs, timing and per-test results accompany the [machine-readable summary](summary.json).

## Original trials and revised offline grades

No model inference was run for this revision. Pinned Harbor's `trials regrade` command restored each original artifact into a separate verifier environment and wrote a new regrade directory. Every original trial directory was hashed before and after and remained unchanged; captured source hashes also match the published submissions. Copied original agent metadata in Harbor's regrade record is provenance, not new inference or billing.

| Saved implementation | Original v0.2.0 trial: passed; reward | v0.3.0 offline regrade: passed; reward |
| --- | ---: | ---: |
| codex-standard-1-replacement | 333/333; 1 | 365/365; 1 |
| codex-standard-2 | 333/333; 1 | 365/365; 1 |
| codex-standard-3 | 333/333; 1 | 357/365; 0 |
| deepseek-standard-1 | 333/333; 1 | 365/365; 1 |
| deepseek-standard-2 | 333/333; 1 | 365/365; 1 |
| deepseek-standard-3 | 333/333; 1 | 365/365; 1 |

The Codex compatibility defect comes from testing the sign of the Python tolerance before NumPy converts it to the risk dtype. A tiny negative scalar can round to negative zero, so equal differences remain ties in the release. Revised failures are functional mismatches, not authentication failures, infrastructure errors, timeouts or changed performance thresholds. Original rewards remain 1; an offline regrade is not an original completed model trial.

To reproduce a saved implementation without any model credentials:

```bash
python3 scripts/regrade_saved.py codex-standard-3 --output jobs/offline-codex-3
```

The public replay command was also executed against the saved Codex submission using the newly built verifier image: native exit 0, 357/365, reward 0, with exactly the same eight failed case IDs as Harbor. See [the replay evidence](public-reproducer/result.json).

The script builds the separate verifier unless an explicit existing image is supplied, executes all cases under the declared resources and preserves CTRF, reward and native command results. Oracle/Nop use the ordinary Harbor commands in the repository README. Diagnostic sources and actual outputs are retained under `diagnostics/`; its `oracle` label denotes the original v0.2.0 reference, archived under `reports/reference-implementations/oracle-v0.2.0`.

## Remaining limitations

The six original standard trials all succeeded. Revised saved-artifact grading does not produce the requested six reward-0 outcomes, and one numerical compatibility failure does not establish frontier-level difficulty. The blocked Codex adversarial prompt is not retried; its DeepSeek counterpart remains withheld, so neither adversarial result is available.

The instruction and four required README explanation paragraphs are AI-generated; they still require the contributor's own authorship. Fluent wording, contributor approval and this engineering repair do not satisfy that requirement. The original substituted rubric review remains 33 pass, 2 fail and 1 not applicable, not an official approval. This revision has a manual delta review. Its old schema flag does not justify changing `network_mode`: the pinned Harbor model recognizes and parses `public`, and the primary-source parser evidence is preserved. No official upstream hosted review or PR CI result is claimed.

Public execution logs are sanitized text exports with LF line endings and trailing display whitespace removed. Private originals, source hashes and all originally published trial records are preserved unchanged.
