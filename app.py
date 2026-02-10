"""
NEXUS - HuggingFace Spaces Entry Point

Launches the Streamlit demo for the NEXUS Maternal-Neonatal Care Platform.
Built with Google HAI-DEF models for the MedGemma Impact Challenge 2026.
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure src/ is on the Python path for imports
ROOT = Path(__file__).parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Set environment defaults for HF Spaces
os.environ.setdefault("STREAMLIT_SERVER_PORT", "7860")
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")


def main():
    app_path = SRC_DIR / "demo" / "streamlit_app.py"
    if not app_path.exists():
        print(f"ERROR: Streamlit app not found at {app_path}")
        sys.exit(1)

    port = os.environ.get("PORT", os.environ["STREAMLIT_SERVER_PORT"])
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)

    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run",
                str(app_path),
                f"--server.port={port}",
                f"--server.address={os.environ['STREAMLIT_SERVER_ADDRESS']}",
                f"--server.headless={os.environ['STREAMLIT_SERVER_HEADLESS']}",
                f"--browser.gatherUsageStats={os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS']}",
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Streamlit process exited with code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("ERROR: Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
