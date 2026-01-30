"""
NEXUS - HuggingFace Spaces Entry Point

Launches the Streamlit demo for the NEXUS Maternal-Neonatal Care Platform.
Built with Google HAI-DEF models for the MedGemma Impact Challenge 2026.

Deploy: https://huggingface.co/spaces/Shahabul/nexus
"""

import subprocess
import sys


def main():
    subprocess.run(
        [
            sys.executable, "-m", "streamlit", "run",
            "src/demo/streamlit_app.py",
            "--server.port=7860",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
