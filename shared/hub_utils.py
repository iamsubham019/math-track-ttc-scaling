"""
shared/hub_utils.py

Hugging Face Hub push/download helpers, shared across all three tracks so
JSON checkpoints and logs are pushed/restored the same way everywhere.
Ported from the code track's src/hub_utils.py.

Reads config["hub"]["repo_id"] (NOT the older config["hf_hub"]["repo_id"]
used by this repo's original src/hub_utils.py, which stays in place
unchanged for backward compatibility with the existing logs/checkpoints
backup flow already relied on in earlier rounds). New code — the learned
latent router and anything else that adopts the shared package — should
import from here and rely on config["hub"].
"""

import os
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


def get_hub_api(config: dict) -> HfApi:
    return HfApi()


def ensure_repo(config: dict):
    repo_id = config["hub"]["repo_id"]
    create_repo(
        repo_id=repo_id,
        private=config["hub"].get("private", False),
        exist_ok=True,
        repo_type="model",
    )
    return repo_id


def push_file(local_path: str, config: dict, path_in_repo: str = None):
    api = get_hub_api(config)
    repo_id = ensure_repo(config)
    path_in_repo = path_in_repo or os.path.basename(local_path)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[shared.hub_utils] pushed {local_path} -> {repo_id}/{path_in_repo}")


def download_file(config: dict, path_in_repo: str, local_path: str) -> bool:
    """Download a Hub file if it exists. Returns True when downloaded."""
    try:
        cached = hf_hub_download(
            repo_id=config["hub"]["repo_id"],
            repo_type="model",
            filename=path_in_repo,
        )
    except EntryNotFoundError:
        return False

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(cached, "rb") as src, open(local_path, "wb") as dst:
        dst.write(src.read())
    print(f"[shared.hub_utils] restored {path_in_repo} -> {local_path}")
    return True


def push_directory(local_dir: str, config: dict, path_in_repo: str = None):
    api = get_hub_api(config)
    repo_id = ensure_repo(config)
    api.upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo or os.path.basename(local_dir.rstrip("/")),
        repo_id=repo_id,
        repo_type="model",
    )
