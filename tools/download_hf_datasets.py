#!/usr/bin/env python3
"""Download any Hugging Face dataset with `snapshot_download`.

This script downloads dataset repos into the default Hub cache layout under
`$HF_HUB_CACHE` (or `$HF_HOME/hub` when `HF_HUB_CACHE` is unset).

WARNING:
    `--clean` is destructive. It removes the existing cached dataset snapshot
    for the target repo before downloading again. If you already downloaded the
    dataset successfully, using `--clean` can delete that good copy and force a
    full re-download.

Examples:
    python tools/download_hf_datasets.py lmms-lab/charades_sta
    python tools/download_hf_datasets.py Apollo-LMMs/LongTimeScope2 --use-auth-token
    python tools/download_hf_datasets.py owner/name --revision main
    python tools/download_hf_datasets.py owner/name --clean
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download


CLEAN_WARNING = (
    "WARNING: --clean is destructive and will delete the existing cached dataset snapshot for this repo. "
    "If you already downloaded the dataset successfully, that good copy may be removed and will need to be downloaded again."
)
CLEAN_ABORTED = "Aborted. Existing cache was not deleted. Command cancelled."


def default_hf_home() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def default_hf_hub_cache() -> Path:
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        return Path(hf_hub_cache).expanduser()
    return default_hf_home() / "hub"


def repo_cache_dir(repo_id: str) -> Path:
    owner, name = repo_id.split("/", 1)
    return default_hf_hub_cache() / f"datasets--{owner}--{name}"


def resolve_token(token_override: str | None, use_auth_token: bool | None) -> str | bool | None:
    if token_override is not None:
        return token_override
    if use_auth_token is True:
        return True
    if use_auth_token is False:
        return False
    return os.environ.get("HF_TOKEN")


def remove_existing_cache(repo_id: str) -> None:
    # This is intentionally destructive: it wipes the cached dataset snapshot.
    repo_dir = repo_cache_dir(repo_id)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)


def confirm_clean(repo_id: str) -> bool:
    repo_dir = repo_cache_dir(repo_id)

    print(CLEAN_WARNING, file=sys.stderr, flush=True)
    print(f"target_cache_dir={repo_dir}", file=sys.stderr, flush=True)
    print(f"cache_exists={'yes' if repo_dir.exists() else 'no'}", file=sys.stderr, flush=True)

    while True:
        try:
            print("Continue and delete the cached dataset snapshot? [y/N]: ", end="", file=sys.stderr, flush=True)
            response = sys.stdin.readline()
            if response == "":
                raise EOFError
            response = response.strip().lower()
        except EOFError:
            print(CLEAN_ABORTED, file=sys.stderr, flush=True)
            return False
        except KeyboardInterrupt:
            print("\n" + CLEAN_ABORTED, file=sys.stderr, flush=True)
            return False

        if response in {"y", "yes"}:
            return True
        if response in {"", "n", "no"}:
            print(CLEAN_ABORTED, file=sys.stderr, flush=True)
            return False

        print("Please answer with 'y' or 'n'.", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download any Hugging Face dataset into the default Hugging Face cache path.",
        epilog=CLEAN_WARNING,
    )
    parser.add_argument("repo_id", help="Hugging Face dataset repo id (`owner/name`).")
    parser.add_argument("--revision", help="Optional revision or commit.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="DANGEROUS: after a y/n confirmation prompt, delete the existing cached repo first. This can remove an already-downloaded good copy and force a full re-download.",
    )
    parser.add_argument("--token", help="Optional HF token override.")
    parser.set_defaults(use_auth_token=None)
    parser.add_argument("--use-auth-token", action="store_true", dest="use_auth_token", help="Use the locally cached Hugging Face token.")
    parser.add_argument("--no-auth-token", action="store_false", dest="use_auth_token", help="Do not use any cached Hugging Face token.")
    parser.add_argument("--max-workers", type=int, default=1, help="snapshot_download max_workers (default: 1).")
    parser.add_argument("--etag-timeout", type=int, default=120, help="ETag timeout seconds passed to snapshot_download.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    token = resolve_token(args.token, args.use_auth_token)

    if args.clean:
        if not confirm_clean(args.repo_id):
            return 1
        remove_existing_cache(args.repo_id)

    print(f"repo_id={args.repo_id}", flush=True)
    print(f"revision={args.revision if args.revision is not None else 'default'}", flush=True)
    print(f"hf_home={default_hf_home()}", flush=True)
    print(f"hf_hub_cache={default_hf_hub_cache()}", flush=True)

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        token=token,
        max_workers=args.max_workers,
        etag_timeout=args.etag_timeout,
        resume_download=True,
    )

    snapshot_dir = Path(snapshot_path)
    print(f"snapshot_dir={snapshot_dir}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
