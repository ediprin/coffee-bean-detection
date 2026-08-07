from pathlib import Path

import pytest

from coffee_detector.hf_sync import HuggingFaceSync


class FakeApi:
    def __init__(self) -> None:
        self.created = []
        self.uploaded = []

    def create_repo(self, **kwargs):
        self.created.append(kwargs)

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)


def test_sync_creates_private_dataset_and_uploads_checkpoint(tmp_path: Path) -> None:
    api = FakeApi()
    run = tmp_path / "A2_seed42"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "last.pt").write_bytes(b"checkpoint")

    sync = HuggingFaceSync(
        "researcher/coffee-vadcp", api=api, path_prefix="screen-v10"
    )
    assert sync.sync_run(run, epoch=3)

    assert api.created == [
        {
            "repo_id": "researcher/coffee-vadcp",
            "repo_type": "dataset",
            "private": True,
            "exist_ok": True,
        }
    ]
    assert api.uploaded[0]["path_in_repo"] == "screen-v10/runs/A2_seed42"
    assert api.uploaded[0]["commit_message"] == "sync A2_seed42: epoch 3"
    assert "weights/last.pt" in api.uploaded[0]["allow_patterns"]


def test_invalid_repo_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="username/nama-repo"):
        HuggingFaceSync("coffee-vadcp", api=FakeApi())
