# Survival concordance optimization — Terminal-Bench 3 candidate

Optimize the two public scikit-survival 0.28.0 concordance APIs while preserving their numerical and statistical behavior. Version 0.3.0 has 365 graded cases and a separate verifier; the original 333-case task remains available at `baseline-v0.2.0`.

Current task revision SHA-256: `df90c60c5d428a34fcc24fbb493655d38bee8d56ea6505ab62c511db3f7b2c4e`. See [the file manifest](reports/task-manifest.json), [task instructions](tasks/survival-concordance/instruction.md), and [task explanations](tasks/survival-concordance/README.md).

Repository attributes preserve file bytes so that Windows Git line-ending settings cannot rewrite the pinned task or execution scripts. The final package is checked against the committed tree and the exact task snapshot used by Harbor.

## Current verifier revision

Version 0.3.0 adds 32 regressions for the already specified NumPy scalar-tolerance and warning behavior, plus a minimal reference warning fix. See the [revision report](reports/verifier-revision/report.md) for actual control results, commands and the six saved-artifact regrades; original trial rewards and trajectories remain unchanged. These are offline regrades with no new model inference. The task instruction, statistical contract, numeric tolerances, performance limits and resources are unchanged.

The [preserved baseline](https://github.com/bennyzry/tb3-survival-concordance/tree/baseline-v0.2.0) remains usable. Unavailable adversarial results and insufficient evidence of frontier-level difficulty remain explicit concerns.

## Genuine final-version model trials

The [final-version evaluation report](reports/final-version-trials/report.md) records new independent standard agent executions against the exact published v0.3.0 commit. These runs are separate from the six original v0.2.0 trials and the six v0.3.0 offline regrades. Native results, complete trajectories, submitted sources, model/effort evidence and CTRF reports accompany the report; unfinished or invalid attempts are not counted as model failures.

## Reproduce

Use x86_64 Linux, Docker, Python 3.12+ and uv. The task containers use a digest-pinned Python 3.13.7 base and pinned dependencies; the supplied native agent bootstrap targets Linux x86_64.

```bash
uv tool install 'harbor==0.23.1.dev202609170426'
docker build --no-cache -t survival-concordance-agent tasks/survival-concordance/environment
docker build --no-cache -t survival-concordance-verifier tasks/survival-concordance/tests
harbor run --path tasks/survival-concordance --agent oracle --n-concurrent 1 --max-retries 0
harbor run --path tasks/survival-concordance --agent nop --n-concurrent 1 --max-retries 0
```

Each task/verifier container is configured for one CPU and 2 GiB RAM. Agent timeout is 14,400 seconds; verifier timeout is 1,200 seconds. Avoid oversubscribing the host. Dependencies come from ordinary public package sources; local wheel caches and undocumented images are not required.

## Reproduce a standard model trial

The pinned native CLI setup avoids Harbor's Node/npm APT bootstrap, which encountered package-download failures on the development host. It downloads official npm artifacts, checks registry integrity, and preserves Codex's complete helper layout. It does not install tools globally.

```bash
python3 scripts/setup_agents.py --directory "$HOME/.cache/tb3-survival-agents"
# Point CODEX_AUTH_JSON_PATH to your existing private ChatGPT auth.json.
python3 scripts/run_standard.py codex --id codex-reproduction-1
# DeepSeek uses the protected ~/.config/tb3/deepseek.key file (mode 600).
python3 scripts/run_standard.py deepseek --id deepseek-reproduction-1
```

Use fresh trial IDs. Keep generated `jobs/` files private; they are ignored by Git. The local DeepSeek adapter forwards unchanged inference bodies to the official TLS endpoint and records model/effort/status metadata without logging the key or message bodies. Agent CLI mounts and the adapter are removed before the separate verifier runs. The scripts do not purchase credits or alter billing or global Codex configuration.

Adversarial evaluation is currently unavailable: the official adversarial prompt was rejected by the Codex provider's safety check, and the same prompt was withheld from DeepSeek rather than rerouted. No adversarial robustness pass or valid adversarial model failure is claimed.

## Preserved v0.2.0 validation

| Implementation | Passed | Failed | Reward |
| --- | ---: | ---: | ---: |
| Oracle | 333 | 0 | 1 |
| Independent alternative | 333 | 0 | 1 |
| Nop (unchanged baseline) | 328 | 5 | 0 |

Both Docker images completed public-source cold builds. These are prior executed validation results, not model-trial results. [CTRF reports](reports/ctrf) and [validation metadata](reports/validation.json) preserve the observations. The [evaluation report](reports/evaluation.md) and [machine-readable status](reports/evaluation.json) report actual model runs when available; pending, infrastructure and authentication failures are not valid task failures.

The later [fresh-checkout validation](reports/packaging-portability.json) records image builds, all 28 static checks and new Oracle/Nop controls after the Git line-ending repair. The earlier results above remain preserved; the later CTRF reports retain their actual failure counts rather than forcing them to match an older timing run.

A standalone, ungraded diagnostic found a compatibility defect in the reward-1 `codex-standard-3` submission for a tiny negative tolerance and equal float16 risks. The v0.2.0 verifier did not detect it, and that baseline remains unchanged. [The original evaluation report](reports/evaluation.md) preserves the fixture and reward 1; the separate v0.3.0 offline regrade now detects the broader defect and receives reward 0 without altering the original trial result.

## Evaluation and compliance

The intended matrix is three Codex / GPT-6.1 Sol / xhigh runs, three Claude Code / DeepSeek V4.1 Flash / max runs, and one adversarial run per configuration. Do not interpret an empty or unfinished result as a model failure.

The instruction and four task explanations were generated by Codex with contributor authorization. They do **not** satisfy the upstream human-authorship requirement; this remains a submission concern. The evidence includes a manual source assessment and a substituted Codex review using the pinned upstream rubric; neither is an official upstream verdict. See the [review follow-up](reports/review-followup.json) for the raw-review limitations and the parser evidence concerning its schema flag. No frontier-model difficulty or novelty claim is established by Oracle/Nop validation.

Author metadata comes from the contributor's existing Git configuration and connected GitHub account. No professional survival-analysis expertise or personal implementation claim is implied.

## License

The supplied scikit-survival source and modifications are GPL-3.0-or-later. Original notices are retained; see [LICENSE](LICENSE), [NOTICE](NOTICE), and the [source/license section](tasks/survival-concordance/README.md#source-and-license).
