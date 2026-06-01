#!/usr/bin/env python3
"""Fetch an arXiv paper as LaTeX source (preferred) or PDF (fallback).

Usage:
    uv run scripts/fetch_paper.py 2410.21276
    uv run scripts/fetch_paper.py https://arxiv.org/abs/2410.21276
    uv run scripts/fetch_paper.py https://arxiv.org/pdf/2410.21276v1
    uv run scripts/fetch_paper.py 2410.21276 --pdf-only

LaTeX source preserves equations cleanly, which matters for any paper
where the maths is load-bearing. PDF is a fallback for papers whose
source isn't on arXiv or for which the e-print endpoint fails.
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
UA = "Mozilla/5.0 (paper-fetch script; +https://arxiv.org)"


def extract_arxiv_id(s: str) -> str:
    m = ARXIV_ID_RE.search(s)
    if not m:
        sys.exit(f"Could not extract arXiv ID from: {s!r}")
    return m.group(1)


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r, dest.open("wb") as f:
        f.write(r.read())


def fetch_latex_source(arxiv_id: str, dest_dir: Path) -> bool:
    """Fetch and extract LaTeX source into dest_dir. Returns True on success."""
    tarball = dest_dir.parent / f"{arxiv_id}.src"
    try:
        _download(f"https://arxiv.org/e-print/{arxiv_id}", tarball)
    except Exception as e:
        print(f"LaTeX source fetch failed: {e}", file=sys.stderr)
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(tarball):
        with tarfile.open(tarball, "r:*") as t:
            try:
                t.extractall(dest_dir, filter="data")
            except TypeError:
                t.extractall(dest_dir)
    else:
        (dest_dir / "main.tex").write_bytes(tarball.read_bytes())
    tarball.unlink(missing_ok=True)
    return True


def fetch_pdf(arxiv_id: str, out_root: Path) -> Path:
    out = out_root / f"{arxiv_id}.pdf"
    _download(f"https://arxiv.org/pdf/{arxiv_id}", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="arXiv ID, abs URL, or pdf URL")
    ap.add_argument("--out", default="papers", help="output dir (default: papers/)")
    ap.add_argument("--pdf-only", action="store_true", help="skip LaTeX, PDF only")
    args = ap.parse_args()

    arxiv_id = extract_arxiv_id(args.source)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.pdf_only:
        latex_dir = out_root / arxiv_id
        if fetch_latex_source(arxiv_id, latex_dir):
            print(f"LaTeX source: {latex_dir}/")
            return
        print("Falling back to PDF.", file=sys.stderr)

    print(f"PDF: {fetch_pdf(arxiv_id, out_root)}")


if __name__ == "__main__":
    main()
