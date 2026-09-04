# Locked-down image for running UNTRUSTED, LLM-generated strategy code.
#
# The container runs the full validation gauntlet on a candidate strategy and
# writes a JSON report. It is launched (see algotrader/agent/sandbox.py) with:
#   --network none        (no exfiltration / no callbacks)
#   --read-only           (immutable root fs; only the mounted /work is writable)
#   --cap-drop ALL        (no Linux capabilities)
#   --security-opt no-new-privileges
#   --memory / --cpus / --pids-limit   (resource caps)
#   --user <uid:gid>      (non-root, matches host so /work stays writable)
#
# Rebuild whenever the algotrader package changes:  bash sandbox/build.sh
FROM python:3.12-slim

# Minimal runtime deps for the gauntlet (no scipy needed — see deflated_sharpe.py).
RUN pip install --no-cache-dir pandas numpy pyarrow pyyaml

WORKDIR /app
# Copy only what the gauntlet needs to run. No secrets, no .env, no network libs.
COPY algotrader /app/algotrader
COPY sandbox/runner.py /app/runner.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Inputs (code, data, config) and the output report all live under /work,
# which the host bind-mounts. Everything else is read-only.
ENTRYPOINT ["python", "/app/runner.py"]
