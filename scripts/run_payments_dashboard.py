"""Launch the INOEMA dashboard (opens in the Payments workspace, which is now the default).

  python -m scripts.run_payments_dashboard [--demo] [--port 8501]

--demo starts in Demonstration mode (persisted, genuine history; live agent/Sandbox actions disabled).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--port", default="8501")
    a = ap.parse_args()
    env = {**os.environ, "INOEMA_DEFAULT_WORKSPACE": "payments", **({"INOEMA_DEMO_MODE": "1"} if a.demo else {})}
    try:
        from spikes.envutil import load_env
        load_env()
        env = {**os.environ, **{k: v for k, v in env.items() if k.startswith("INOEMA_")}}
    except Exception:  # noqa: BLE001
        pass
    return subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "dashboard", "app.py"), "--server.port", a.port], cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
