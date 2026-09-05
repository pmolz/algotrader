"""Host status card: liveness, model residency, GPU telemetry.

The state this card exists for is the quiet one. Under `host: auto` a machine
that drops off the network fails nothing — the session moves to the fallback and
the experiment log looks like any other night. So "up but missing the model" and
"up but nothing loaded in VRAM" have to be distinguishable from "healthy", or the
card is decoration.
"""

from __future__ import annotations

import pytest

from algotrader.dashboard import hoststatus


@pytest.fixture(autouse=True)
def clear_cache():
    hoststatus._cache.update(at=0.0, value=None)


def cfg(**over):
    base = {
        "agent": {
            "model": "qwen2.5-coder:7b",
            "host": "auto",
            "host_preference": ["laptop", "local"],
            "hosts": {
                "local": {"url": "http://127.0.0.1:11434", "gpu": "local"},
                "laptop": {"url": "http://10.0.0.249:11434", "gpu": "ssh:dell"},
            },
        }
    }
    base["agent"].update(over)
    return base


def fake_probes(monkeypatch, states: dict, gpu=None):
    """states maps url -> (up, models, resident)."""
    def probe(url):
        up, models, resident = states[url]
        return {"up": up, "error": None if up else "URLError: refused",
                "latency_ms": 12 if up else None, "models": models,
                "resident": resident}
    monkeypatch.setattr(hoststatus, "probe_endpoint", probe)
    monkeypatch.setattr(hoststatus, "gpu_stats", lambda src: gpu)


LOCAL, LAPTOP = "http://127.0.0.1:11434", "http://10.0.0.249:11434"
GPU_OK = {"error": None, "name": "RTX A2000", "util_pct": 43.0,
          "mem_used_mb": 4650.0, "mem_total_mb": 8192.0, "mem_pct": 56.8,
          "temp_c": 71.0, "power_w": 34.0}


def test_healthy_hosts_report_up_and_the_preferred_one_is_selected(monkeypatch):
    fake_probes(monkeypatch, {
        LOCAL: (True, ["qwen2.5-coder:7b"], None),
        LAPTOP: (True, ["qwen2.5-coder:7b"], {"name": "qwen2.5-coder:7b", "vram_mb": 4528}),
    }, gpu=GPU_OK)
    out = hoststatus.collect(cfg())
    assert {h["name"]: h["state"] for h in out["hosts"]} == {"laptop": "up", "local": "up"}
    assert out["selected"] == "laptop"          # first in host_preference
    assert out["hosts"][0]["gpu"]["temp_c"] == 71.0


def test_a_reachable_host_without_the_model_is_not_merely_up(monkeypatch):
    """It is online and still cannot serve this agent. Different problem,
    different fix, so it gets its own state rather than a green dot."""
    fake_probes(monkeypatch, {
        LOCAL: (True, ["qwen2.5-coder:7b"], None),
        LAPTOP: (True, ["llama3:8b"], None),
    })
    out = hoststatus.collect(cfg())
    by = {h["name"]: h for h in out["hosts"]}
    assert by["laptop"]["state"] == "no-model"
    assert by["laptop"]["up"] is True and by["laptop"]["has_model"] is False
    assert out["selected"] == "local", "a host without the model must not be chosen"


def test_selection_mirrors_a_pinned_policy(monkeypatch):
    fake_probes(monkeypatch, {
        LOCAL: (True, ["qwen2.5-coder:7b"], None),
        LAPTOP: (False, [], None),
    })
    # pinned to a host that is down -> nothing selected, no silent fallback
    assert hoststatus.collect(cfg(host="laptop"))["selected"] is None
    hoststatus._cache.update(at=0.0, value=None)
    assert hoststatus.collect(cfg(host="local"))["selected"] == "local"


def test_no_usable_host_selects_nothing(monkeypatch):
    fake_probes(monkeypatch, {LOCAL: (False, [], None), LAPTOP: (False, [], None)})
    out = hoststatus.collect(cfg())
    assert out["selected"] is None
    assert all(h["state"] == "down" for h in out["hosts"])


def test_gpu_is_not_probed_on_a_host_that_is_down(monkeypatch):
    """No point paying an SSH timeout for a box that just failed an HTTP probe."""
    calls = []
    fake_probes(monkeypatch, {LOCAL: (True, ["qwen2.5-coder:7b"], None),
                              LAPTOP: (False, [], None)})
    monkeypatch.setattr(hoststatus, "gpu_stats", lambda src: calls.append(src) or GPU_OK)
    hoststatus.collect(cfg())
    assert "ssh:dell" not in calls


def test_results_are_cached_so_polling_does_not_spawn_ssh_per_request(monkeypatch):
    probes = []
    def probe(url):
        probes.append(url)
        return {"up": True, "error": None, "latency_ms": 1,
                "models": ["qwen2.5-coder:7b"], "resident": None}
    monkeypatch.setattr(hoststatus, "probe_endpoint", probe)
    monkeypatch.setattr(hoststatus, "gpu_stats", lambda src: None)
    for _ in range(5):
        hoststatus.collect(cfg())
    assert len(probes) == 2, f"expected one probe per host, got {len(probes)}"


def test_broken_host_config_is_reported_not_raised(monkeypatch):
    out = hoststatus.collect({"agent": {"hosts": "not-a-mapping", "model": "m"}})
    assert out["hosts"] == [] and out["error"]


# -- nvidia-smi parsing -------------------------------------------------------
def test_gpu_output_is_parsed(monkeypatch):
    line = "NVIDIA RTX A2000 8GB Laptop GPU, 43, 4650, 8192, 71, 34.21\n"
    monkeypatch.setattr(hoststatus, "_run_nvidia_smi", lambda src: (line, None))
    g = hoststatus.gpu_stats("local")
    assert (g["util_pct"], g["mem_used_mb"], g["temp_c"], g["power_w"]) == (43.0, 4650.0, 71.0, 34.21)
    assert g["mem_pct"] == 56.8


def test_power_reported_as_not_available_does_not_break_parsing(monkeypatch):
    """Plenty of laptop GPUs return [N/A] for power draw."""
    monkeypatch.setattr(hoststatus, "_run_nvidia_smi",
                        lambda src: ("RTX A2000, 12, 100, 8192, 44, [N/A]\n", None))
    g = hoststatus.gpu_stats("local")
    assert g["power_w"] is None and g["util_pct"] == 12.0


def test_gpu_errors_surface_instead_of_disappearing(monkeypatch):
    monkeypatch.setattr(hoststatus, "_run_nvidia_smi", lambda src: (None, "timed out"))
    assert hoststatus.gpu_stats("ssh:dell")["error"] == "timed out"


def test_no_gpu_source_means_no_gpu_section():
    assert hoststatus.gpu_stats(None) is None


def test_unknown_gpu_source_is_rejected():
    assert "unknown gpu source" in hoststatus.gpu_stats("carrier-pigeon")["error"]
