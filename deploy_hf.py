"""
Deploy NEXUS to HuggingFace Spaces.

Prerequisites:
1. Create a Space at https://huggingface.co/new-space
   - Owner: Shahabul, Name: nexus, SDK: Docker, Public
2. Create a write token at https://huggingface.co/settings/tokens
   - Type: Write, or Fine-grained with repo write permissions
3. Run: HF_TOKEN=hf_xxx python deploy_hf.py
"""
import os
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

SPACE_ID = "Shahabul/nexus"
PROJECT_ROOT = Path(__file__).parent

def deploy():
    api = HfApi()
    
    # Create a staging directory with only the files needed
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        
        # Core files
        for f in ["Dockerfile", "README.md", "requirements_spaces.txt", "app.py"]:
            shutil.copy2(PROJECT_ROOT / f, staging / f)
        
        # Source code
        shutil.copytree(PROJECT_ROOT / "src", staging / "src")
        
        # Model metadata (JSON configs, not weights)
        (staging / "models" / "linear_probes").mkdir(parents=True)
        for json_file in (PROJECT_ROOT / "models" / "linear_probes").glob("*.json"):
            shutil.copy2(json_file, staging / "models" / "linear_probes" / json_file.name)
        
        # Pre-computed embeddings for edge mode
        if (PROJECT_ROOT / "models" / "embeddings").exists():
            shutil.copytree(
                PROJECT_ROOT / "models" / "embeddings",
                staging / "models" / "embeddings"
            )
        
        print(f"Uploading to {SPACE_ID}...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=SPACE_ID,
            repo_type="space",
        )
        print(f"Deployed! Visit: https://huggingface.co/spaces/{SPACE_ID}")

if __name__ == "__main__":
    deploy()
