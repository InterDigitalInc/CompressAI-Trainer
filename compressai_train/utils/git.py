import os
from shlex import quote
from typing import Iterable, Optional


def branch_name(rev: str = "HEAD", root: str = ".") -> str:
    cmd = f"git -C {quote(root)} rev-parse --abbrev-ref {quote(rev)}"
    return os.popen(cmd).read().rstrip()


def common_ancestor_hash(
    rev1: str = "HEAD", rev2: Optional[str] = None, root: str = "."
) -> str:
    if rev2 is None:
        rev2 = main_branch_name(root=root)
    if branch_name(rev1, root=root) == branch_name(rev2, root=root):
        return commit_hash(rev=rev1, root=root)
    cmd = (
        "diff -u "
        f"<(git -C {quote(root)} rev-list --first-parent {quote(rev1)}) "
        f"<(git -C {quote(root)} rev-list --first-parent {quote(rev2)}) | "
        "sed -ne 's/^ //p' | head -1"
    )
    return os.popen(cmd).read().rstrip()


def commit_hash(rev: str = "HEAD", short: bool = False, root: str = ".") -> str:
    options = "--short" if short else ""
    cmd = f"git -C {quote(root)} rev-parse {options} {quote(rev)}"
    return os.popen(cmd).read().rstrip()


def diff(rev: str = "HEAD", root: str = ".") -> str:
    cmd = f"git -C {quote(root)} --no-pager diff --no-color {quote(rev)}"
    return os.popen(cmd).read().rstrip()


def main_branch_name(
    root: str = ".", candidates: Iterable[str] = ("main", "master")
) -> str:
    r"""Returns name of primary branch (main or master)."""
    candidates_str = " ".join(quote(x) for x in candidates)
    cmd = f"git -C {quote(root)} branch -l {candidates_str}"
    lines = os.popen(cmd).read().rstrip().splitlines()
    lines = [_removeprefix(x, "* ").strip() for x in lines]
    assert len(lines) == 1
    return lines[0]


def _removeprefix(s: str, prefix: str) -> str:
    return s[len(prefix) :] if s.startswith(prefix) else s
