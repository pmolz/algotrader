#!/usr/bin/env bash
# Set up a machine to serve Ollama to the rest of the LAN.
#
# Copy this to the box that will host the model and run it there:
#     scp scripts/setup_llm_host.sh <laptop>:
#     ssh <laptop> 'bash setup_llm_host.sh --subnet 10.0.0.0/24'
#
# Idempotent — safe to re-run after changing the model or the subnet.
#
# What it does, and why:
#   * binds Ollama to 0.0.0.0 (it listens on localhost only by default)
#   * keeps the model resident, so an overnight loop doesn't re-pay the load
#     cost on every single iteration
#   * opens 11434 to one subnet only — Ollama has NO authentication
#   * stops the lid closing from killing the server mid-session
set -euo pipefail

SUBNET="${SUBNET:-10.0.0.0/24}"
MODEL="${MODEL:-qwen2.5-coder:7b}"
KEEP_ALIVE="${KEEP_ALIVE:-8h}"
PORT=11434

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subnet) SUBNET="$2"; shift 2 ;;
        --model)  MODEL="$2";  shift 2 ;;
        --keep-alive) KEEP_ALIVE="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }

# -- 1. GPU ------------------------------------------------------------------
say "Checking the GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [[ "$VRAM_MB" -lt 6000 ]]; then
        warn "only ${VRAM_MB}MiB of VRAM — a 7B model at Q4 needs ~5GB plus context."
        warn "Consider --model qwen2.5-coder:3b, or expect slow CPU offload."
    fi
    # Persistence mode keeps the driver initialised so the first request after
    # an idle stretch doesn't pay driver startup.
    sudo nvidia-smi -pm 1 >/dev/null 2>&1 || warn "could not enable persistence mode"
else
    warn "nvidia-smi not found — Ollama will run on CPU, which is far too slow"
    warn "for an overnight loop. Install the NVIDIA driver first."
fi

# -- 2. Ollama ---------------------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    say "Ollama already installed ($(ollama --version 2>&1 | head -1))"
elif command -v pacman >/dev/null 2>&1; then
    # Arch-family: prefer the repo package. It carries the right CUDA deps and
    # stays current with pacman, where the upstream installer drops an
    # unmanaged tree in /usr/local that you have to re-run to update.
    say "Installing ollama-cuda from the distro repos"
    sudo pacman -S --needed --noconfirm ollama-cuda
else
    say "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
fi

# -- 3. Listen on the LAN ----------------------------------------------------
say "Configuring the service to listen on 0.0.0.0:${PORT}"
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOF
[Service]
# Default is 127.0.0.1 — without this nothing off-box can reach it.
Environment="OLLAMA_HOST=0.0.0.0:${PORT}"
# Keep the model in VRAM between requests. The default 5m unloads it between
# iterations of a slow loop, re-paying tens of seconds of load each time.
Environment="OLLAMA_KEEP_ALIVE=${KEEP_ALIVE}"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
sudo systemctl restart ollama

# -- 4. Firewall -------------------------------------------------------------
# Ollama has no auth: anyone who can reach the port can run anything you have
# pulled. Scope it to one subnet.
say "Opening port ${PORT} to ${SUBNET} only"
if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
    sudo ufw allow from "$SUBNET" to any port "$PORT" proto tcp
elif command -v firewall-cmd >/dev/null 2>&1 && sudo firewall-cmd --state >/dev/null 2>&1; then
    sudo firewall-cmd --permanent --add-rich-rule="rule family=ipv4 source address=${SUBNET} port port=${PORT} protocol=tcp accept"
    sudo firewall-cmd --reload
else
    warn "no active ufw/firewalld found. If this machine ever joins an untrusted"
    warn "network, restrict port ${PORT} yourself — it is unauthenticated."
fi

# -- 5. Survive the lid ------------------------------------------------------
say "Keeping the machine awake with the lid closed"
sudo mkdir -p /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/99-ollama-host.conf >/dev/null <<'EOF'
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
EOF
# NOTE: do *not* restart systemd-logind here. Restarting it on a live system
# orphans the running graphical session — the dead session keeps its claim on
# seat0, every new login then fails at pam_open_session, and the user is bounced
# straight back to the greeter with a password that was actually correct. The
# drop-in above is read at boot, so a reboot is both sufficient and safe.
NEEDS_REBOOT=1
# Suspend would drop the session mid-generation.
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target >/dev/null 2>&1 \
    || warn "could not mask sleep targets"

# -- 6. Model ----------------------------------------------------------------
say "Pulling ${MODEL}"
ollama pull "$MODEL"

# -- 7. Verify ---------------------------------------------------------------
say "Verifying"
IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
sleep 1
if curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null; then
    echo "  local API:  OK"
else
    echo "  local API:  FAILED — check: journalctl -u ollama -n 50" >&2
    exit 1
fi
echo "  models:     $(ollama list | tail -n +2 | awk '{print $1}' | paste -sd, -)"
echo "  listening:  $(ss -ltnp 2>/dev/null | grep ":${PORT}" | awk '{print $4}' | paste -sd, - || echo '?')"

cat <<EOF

$(printf '\033[1mDone.\033[0m') This machine is reachable at: http://${IP:-<ip>}:${PORT}
${NEEDS_REBOOT:+
$(printf '\033[33mReboot before relying on this.\033[0m') The lid/sleep settings are read at
boot, and rebooting is the only safe way to apply them — restarting logind on a
live system breaks graphical logins until the next boot.
}

On the desktop, put this in config/local.yaml:

  agent:
    hosts:
      laptop:
        url: http://${IP:-<ip>}:${PORT}
        model: ${MODEL}

then confirm it from there with:

  python scripts/check_llm.py
EOF
