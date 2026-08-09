#!/usr/bin/env python3
"""Materialize the pinned upstream checkouts that ``build_registry.py`` reads.

``scripts/build_registry.py`` is the sole producer of ``data/scores.csv``, and
it expects one directory per upstream repository below ``--sources`` (default
``/tmp/pathopress_sources``).  Those checkouts are not vendored in this
repository, so without this script a reader can rerun every downstream analysis
but cannot rebuild the registry input.  This script closes that gap.

Every revision comes from ``data/provenance.json`` -- the same file the build
writes back -- so no branch name is ever followed.  Each checkout is verified by
``git rev-parse HEAD`` after the fetch and a mismatch is a hard error.  The
committed ``source_data/*.csv`` snapshots are verified against the SHA-256
digests recorded in the same file.

Like ``build_registry.py`` this uses the Python standard library only.

Examples
--------
    python3 scripts/fetch_sources.py --dry-run     # resolve pins, fetch nothing
    python3 scripts/fetch_sources.py               # materialize the checkouts
    python3 scripts/fetch_sources.py --check       # verify an existing tree
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = Path("/tmp/pathopress_sources")
DEFAULT_PROVENANCE = REPOSITORY_ROOT / "data" / "provenance.json"

# Sources needed only to regenerate a committed ``source_data/*.csv`` snapshot,
# never by ``build_registry.py`` itself.  ``data/provenance.json`` does not pin
# them, so the revision below is the checkout this project actually used and is
# recorded here rather than being silently floated to a branch tip.
EXTRACTOR_ONLY_REPOSITORIES = {
    "eva_openmidnight": {
        "url": "https://github.com/MedARC-AI/OpenMidnight",
        "commit": "4c3e4a83802010f47dc68bb2d25629f2b6f58eea",
        "consumer": "scripts/extract_group_b_official_scores.py",
        "pinned_in_provenance": False,
    },
}


# Byte-exact copies of upstream files that this repository vendors under
# ``source_data/pinned/``. Comparing a fresh checkout against them is a
# content-level integrity check that does not depend on trusting git alone.
# The upstream paths mirror ``PINNED_SOURCE_DESTINATIONS`` in
# ``scripts/build_score_review_ledger.py``.
PINNED_FILE_CHECKS = {
    "eva/tools/data/leaderboards/pathology.csv": "source_data/pinned/eva/pathology.csv",
    "eva_midnight/README.md": "source_data/pinned/eva_midnight/README.md",
    "thunder/docs/leaderboards.md": "source_data/pinned/thunder/leaderboards.md",
    "hest/README.md": "source_data/pinned/hest/README.md",
    "pathorob/README.md": "source_data/pinned/pathorob/README.md",
}


class PinError(RuntimeError):
    """Raised when a checkout does not match its recorded revision."""


def git_env() -> dict[str, str]:
    """Environment that never prompts for credentials and never pulls LFS blobs.

    Every file the registry parses is plain text in the git tree.  Smudging LFS
    pointers would download model weights (~1 GB for the Midnight model card's
    repository) that nothing here reads.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    env.setdefault("GIT_ASKPASS", "true")
    return env


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        env=git_env(),
    )
    if check and result.returncode != 0:
        raise PinError(
            f"git {' '.join(args)} failed in {repo}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def head_commit(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        return git(repo, "rev-parse", "HEAD")
    except PinError:
        return None


def load_pins(provenance_path: Path, include_extractor_sources: bool) -> dict[str, dict]:
    document = json.loads(provenance_path.read_text())
    repositories = document.get("repositories")
    if not repositories:
        raise PinError(f"{provenance_path} records no 'repositories' block")
    pins: dict[str, dict] = {}
    for name, entry in repositories.items():
        url = entry.get("url")
        commit = entry.get("commit")
        if not url or not commit:
            raise PinError(f"{provenance_path}: repository '{name}' lacks a url/commit pin")
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit.lower()):
            raise PinError(
                f"{provenance_path}: repository '{name}' pin '{commit}' is not a full "
                "40-character commit SHA; refusing to fetch a branch or tag name"
            )
        pins[name] = {"url": url, "commit": commit, "pinned_in_provenance": True}
    if include_extractor_sources:
        for name, entry in EXTRACTOR_ONLY_REPOSITORIES.items():
            pins[name] = dict(entry)
    return pins


def remote_has_commit(url: str, commit: str) -> bool:
    """True when the exact commit is still fetchable from the remote.

    Tries the cheap ref listing first (the pin is often a branch or tag tip),
    then a blobless, depth-1 probe fetch into a scratch bare repository, which
    transfers only commit and tree objects.
    """
    listing = subprocess.run(
        ["git", "ls-remote", url],
        text=True,
        capture_output=True,
        env=git_env(),
    )
    if listing.returncode != 0:
        return False
    if any(line.split("\t", 1)[0] == commit for line in listing.stdout.splitlines()):
        return True

    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        probe = Path(scratch) / "probe.git"
        subprocess.run(
            ["git", "init", "--quiet", "--bare", str(probe)],
            check=False,
            capture_output=True,
            env=git_env(),
        )
        attempt = subprocess.run(
            [
                "git", "-C", str(probe), "fetch", "--quiet",
                "--depth=1", "--filter=blob:none", url, commit,
            ],
            text=True,
            capture_output=True,
            env=git_env(),
        )
        return attempt.returncode == 0


def fetch_commit(repo: Path, url: str, commit: str) -> None:
    """Fetch exactly ``commit`` into ``repo``, falling back to a full clone.

    Both GitHub and the Hugging Face git endpoints serve arbitrary reachable
    SHAs, so the shallow path is the normal one.  A server that refuses it makes
    the fallback necessary; the fallback still checks out the pinned SHA.
    """
    repo.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        git(repo, "init", "--quiet")
    existing = git(repo, "remote", check=False)
    if "origin" in existing.split():
        git(repo, "remote", "set-url", "origin", url)
    else:
        git(repo, "remote", "add", "origin", url)

    shallow = subprocess.run(
        ["git", "-C", str(repo), "fetch", "--quiet", "--depth=1", "origin", commit],
        text=True,
        capture_output=True,
        env=git_env(),
    )
    if shallow.returncode != 0:
        print(
            f"  shallow fetch of {commit[:12]} was refused; falling back to a full fetch",
            file=sys.stderr,
        )
        git(repo, "fetch", "--quiet", "--tags", "origin")
    git(repo, "checkout", "--quiet", "--detach", commit)


def ensure_repository(
    name: str,
    pin: dict,
    destination: Path,
    *,
    dry_run: bool,
    check_only: bool,
) -> str:
    repo = destination / name
    url, commit = pin["url"], pin["commit"]
    current = head_commit(repo)

    if current == commit:
        return "already at pin"

    if repo.exists() and current is None and any(repo.iterdir()):
        raise PinError(
            f"{repo} exists but is not a git checkout. Refusing to overwrite it; "
            "remove it or pass a different --destination."
        )

    if check_only:
        raise PinError(
            f"{name}: expected {commit} but found "
            f"{current or 'no checkout'} at {repo}"
        )

    if dry_run:
        if not remote_has_commit(url, commit):
            raise PinError(
                f"{name}: pinned commit {commit} is not retrievable from {url}. "
                "The upstream history may have been rewritten or the repository "
                "removed; data/scores.csv cannot be rebuilt from this source."
            )
        return "pin resolves on remote (dry run, nothing fetched)"

    fetch_commit(repo, url, commit)
    landed = head_commit(repo)
    if landed != commit:
        raise PinError(
            f"{name}: checkout landed on {landed} but data/provenance.json pins "
            f"{commit}. Not continuing with a mismatched source."
        )
    return "fetched and checked out" if current is None else f"moved from {current[:12]}"


def verify_pinned_files(destination: Path, names: set[str]) -> tuple[int, list[str]]:
    """Compare checked-out files against the byte-exact copies vendored here."""
    problems: list[str] = []
    checked = 0
    for upstream, vendored in PINNED_FILE_CHECKS.items():
        if upstream.split("/", 1)[0] not in names:
            continue
        reference = REPOSITORY_ROOT / vendored
        candidate = destination / upstream
        if not reference.exists():
            problems.append(f"missing vendored reference {vendored}")
            continue
        if not candidate.exists():
            problems.append(f"{upstream}: absent from the checkout at {destination}")
            continue
        checked += 1
        if candidate.read_bytes() != reference.read_bytes():
            problems.append(
                f"{upstream}: checkout differs from the byte-exact copy in {vendored}"
            )
    return checked, problems


def verify_snapshots(provenance_path: Path) -> tuple[int, list[str]]:
    """Verify the committed ``source_data`` snapshots against provenance digests."""
    document = json.loads(provenance_path.read_text())
    problems: list[str] = []
    checked = 0
    for report, entry in document.get("source_reports", {}).items():
        digests = entry.get("snapshot_sha256")
        expected: list[tuple[str, str]] = []
        if isinstance(digests, str) and entry.get("snapshot_path"):
            expected.append((entry["snapshot_path"], digests))
        elif isinstance(digests, dict):
            expected.extend(("source_data/" + name, sha) for name, sha in digests.items())
        for relative, sha in expected:
            path = REPOSITORY_ROOT / relative
            if not path.exists():
                problems.append(f"{report}: missing snapshot {relative}")
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            checked += 1
            if actual != sha:
                problems.append(
                    f"{report}: {relative} digest {actual} != recorded {sha}"
                )
    return checked, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCES,
        help=f"where to materialize the checkouts (default: {DEFAULT_SOURCES}, "
        "which is what scripts/build_registry.py --sources defaults to)",
    )
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve every pin against its remote without cloning anything",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify an existing tree against the pins; fetch nothing, fail on drift",
    )
    parser.add_argument(
        "--include-extractor-sources",
        action="store_true",
        help="also fetch sources needed only to regenerate committed source_data "
        "snapshots (currently OpenMidnight, whose revision is not pinned in "
        "data/provenance.json)",
    )
    parser.add_argument(
        "--skip-snapshot-verification",
        action="store_true",
        help="skip the SHA-256 check of the committed source_data snapshots",
    )
    args = parser.parse_args()

    if args.dry_run and args.check:
        parser.error("--dry-run and --check are mutually exclusive")

    try:
        pins = load_pins(args.provenance, args.include_extractor_sources)
    except PinError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    destination = args.destination.expanduser()
    print(f"provenance : {args.provenance}")
    print(f"destination: {destination}")
    print(f"repositories: {len(pins)}\n")

    failures: list[str] = []
    for name in sorted(pins):
        pin = pins[name]
        label = "" if pin.get("pinned_in_provenance", True) else " [not pinned in provenance.json]"
        print(f"{name} @ {pin['commit'][:12]}{label}")
        try:
            status = ensure_repository(
                name, pin, destination, dry_run=args.dry_run, check_only=args.check
            )
        except PinError as error:
            print(f"  FAIL: {error}", file=sys.stderr)
            failures.append(name)
        else:
            print(f"  {status}")

    if not failures and not args.dry_run:
        checked, problems = verify_pinned_files(destination, set(pins))
        print(f"\nvendored byte-exact upstream files matched: {checked}")
        for problem in problems:
            print(f"  FAIL: {problem}", file=sys.stderr)
        failures.extend(problems)

    if not args.skip_snapshot_verification:
        checked, problems = verify_snapshots(args.provenance)
        print(f"source_data snapshots verified: {checked}")
        for problem in problems:
            print(f"  FAIL: {problem}", file=sys.stderr)
        failures.extend(problems)

    if failures:
        print(
            f"\n{len(failures)} problem(s). data/scores.csv must not be rebuilt "
            "from this tree.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("\nAll pins resolve. Rerun without --dry-run to materialize them.")
    elif args.check:
        print("\nEvery checkout matches data/provenance.json.")
    else:
        print(
            "\nEvery checkout matches data/provenance.json. Next:\n"
            f"  python3 scripts/build_registry.py --sources {destination}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
