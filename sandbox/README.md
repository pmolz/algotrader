# Sandbox — running untrusted generated code safely

The agent asks an LLM to write strategy code, then *executes* it. That is running
untrusted code. In-process `exec` (with a denylist + restricted builtins) is fine
for supervised local runs, but for **unattended / overnight** loops you must
isolate it. This directory provides a Docker jail that runs the whole validation
gauntlet on a candidate strategy with no way to reach your machine.

## Build

```bash
bash sandbox/build.sh          # rebuild whenever algotrader/ changes
```

## Enable

Either in `config/default.yaml`:

```yaml
agent:
  use_sandbox: true
```

or per run:

```bash
python scripts/run_agent.py --symbol BTC/USDT --iterations 10 --sandbox
```

## What's enforced (verified)

| Guarantee            | Flag                              |
|----------------------|-----------------------------------|
| No network at all    | `--network none`                  |
| Read-only root FS    | `--read-only` (+ `--tmpfs /tmp`)  |
| No capabilities      | `--cap-drop ALL`                  |
| No privilege escalation | `--security-opt no-new-privileges` |
| Fork-bomb / OOM / CPU caps | `--pids-limit`, `--memory`, `--cpus` |
| Non-root user        | `--user <host uid:gid>`           |
| Hard wall-clock kill | subprocess `timeout`              |

The only writable shared surface is a per-run temp dir bind-mounted at `/work`
(holds `strategy.py`, `data.parquet`, `params.json`, and receives `report.json`).
It is deleted after each run.

## Flow

```
host: write code+data+params to temp /work  ->  docker run (jailed) runner.py
                                                    loads untrusted strategy
                                                    runs the gauntlet
                                                    writes report.json
host: read report.json  ->  record to experiment DB
```

## Defense in depth

Even inside the jail, `algotrader/agent/codegen.py` still screens code (denylist,
restricted builtins, import whitelist of pandas/numpy/math). So a malicious
strategy has to get past *both* the code screen and the container isolation.
