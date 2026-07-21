from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


ARTIFACT_PATTERNS = [
    "weights/last.pt",
    "weights/best.pt",
    "results.csv",
    "args.yaml",
    "experiment_manifest.json",
    "**/weights/last.pt",
    "**/weights/best.pt",
    "**/results.csv",
    "**/args.yaml",
    "**/experiment_manifest.json",
    "**/combined_view_manifest.json",
    "**/reports/*.json",
    "**/confirmation.json",
]


class HuggingFaceSync:
    """Incrementally persist compact experiment artifacts to the Hub."""

    def __init__(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        repo_type: str = "dataset",
        path_prefix: str = "vadcp-ablation",
        private: bool = True,
        api: Any | None = None,
        retries: int = 2,
    ) -> None:
        if not repo_id or "/" not in repo_id:
            raise ValueError("repo_id Hugging Face harus berbentuk username/nama-repo")
        if repo_type not in {"dataset", "model"}:
            raise ValueError("repo_type harus dataset atau model")
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as error:  # pragma: no cover
                raise RuntimeError(
                    "huggingface_hub belum terpasang; jalankan pip install -e ."
                ) from error
            api = HfApi(token=token or os.getenv("HF_TOKEN"))
        self.api = api
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.path_prefix = path_prefix.strip("/")
        self.retries = max(int(retries), 0)
        self.api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    def _upload(self, folder: Path, path_in_repo: str, message: str) -> bool:
        if not folder.is_dir():
            print(f"[HF SYNC SKIP] folder belum ada: {folder}", flush=True)
            return False
        for attempt in range(self.retries + 1):
            try:
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    folder_path=str(folder),
                    path_in_repo=path_in_repo,
                    allow_patterns=ARTIFACT_PATTERNS,
                    commit_message=message,
                )
                print(
                    f"[HF SYNC OK] {self.repo_id}/{path_in_repo} | {message}",
                    flush=True,
                )
                return True
            except Exception as error:  # do not kill expensive GPU work
                if attempt < self.retries:
                    delay = 2**attempt
                    print(
                        f"[HF SYNC RETRY {attempt + 1}] {type(error).__name__}: "
                        f"{error} | tunggu {delay}s",
                        flush=True,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"[HF SYNC FAILED] {type(error).__name__}: {error}",
                        flush=True,
                    )
        return False

    def sync_run(self, run_dir: str | Path, epoch: int | None = None) -> bool:
        run_dir = Path(run_dir).expanduser().resolve()
        suffix = f"epoch {epoch}" if epoch is not None else "run complete"
        remote = "/".join(filter(None, (self.path_prefix, "runs", run_dir.name)))
        return self._upload(run_dir, remote, f"sync {run_dir.name}: {suffix}")

    def sync_output(self, output_root: str | Path, reason: str) -> bool:
        output_root = Path(output_root).expanduser().resolve()
        remote = "/".join(filter(None, (self.path_prefix, "output")))
        return self._upload(output_root, remote, f"sync experiment: {reason}")
