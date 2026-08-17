"""
Hugging Face Hub helpers: push logs/checkpoints so Kaggle session restarts
don't lose progress, per the guide's checkpointing requirement.

Requires HF_TOKEN to be set in the environment (see README quickstart).
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download


def ensure_repo(cfg: dict):
    api = HfApi()
    repo_id = cfg["hf_hub"]["repo_id"]
    create_repo(repo_id, private=cfg["hf_hub"].get("private", True), exist_ok=True)
    return repo_id


def push_file(local_path: str, path_in_repo: str, cfg: dict):
    repo_id = ensure_repo(cfg)
    upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=os.environ.get("HF_TOKEN"),
    )


def pull_file(path_in_repo: str, local_path: str, cfg: dict):
    """Restore a previously checkpointed file after a session restart."""
    repo_id = cfg["hf_hub"]["repo_id"]
    downloaded = hf_hub_download(repo_id=repo_id, filename=path_in_repo,
                                  token=os.environ.get("HF_TOKEN"))
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    if downloaded != local_path:
        import shutil
        shutil.copy(downloaded, local_path)
    return local_path
