# Colab Shared-Drive Runbook

## Permanent project rule

All Colab accounts must use one shared project root. The logical root is:

```text
/content/drive/MyDrive/Coffee_Bean_Detection
```

DriveFS may expose the same folder internally as:

```text
/content/drive/.shortcut-targets-by-id/<id>/Coffee_Bean_Detection
```

These are two views of one shared folder, not two projects. Never create a new
`Coffee_Bean_Detection` root when discovery fails.

## Recorded incident: 2026-08-02

The Faruq-v3 operational audit repeatedly reported that the dataset archive and
checkpoint were missing even though both existed in the canonical shared root.
The cause was DriveFS shortcut traversal: recursive `Path.rglob()` did not enter
the shortcut, while direct access through
`/content/drive/MyDrive/Coffee_Bean_Detection` worked.

The permanent resolver contract is therefore:

1. probe the direct MyDrive shortcut path first;
2. accept a marker-verified shortcut even when its target has an ID or alias;
3. use exact required artifact paths as the fallback authority;
4. use recursive discovery only after direct probing;
5. never create a replacement project root;
6. force-remount Drive only after a shortcut is added, moved, or restored.

This behavior is implemented in `coffee_detector.drive_project` and protected
by `tests/test_drive_project.py`.

## GPU and quota rule

Storage discovery is a CPU preflight. Before allocating or using a GPU, verify:

- the shared root is visible;
- every required archive/checkpoint exists through its exact relative path;
- the intended output directory belongs to that same root;
- no training or inference process has started.

If this preflight fails, disconnect the GPU runtime. Do not debug Drive paths
while an accelerator remains allocated.

## Switching Google accounts

The owner shares the one canonical folder with Editor access. Every other
account adds a shortcut to that folder under `My Drive`. Do not share or create
separate copies of `bundles`, `checkpoints`, or `experiments`. Completed and
resumable artifacts must remain inside the canonical shared root.

