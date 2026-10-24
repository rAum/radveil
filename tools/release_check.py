# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0

"""Check that a release tag matches the version everywhere, and extract its
release notes from CHANGELOG.md.

    python tools/release_check.py v0.1.0                 # check only
    python tools/release_check.py v0.1.0 --notes notes.md

The release workflow runs this first, so a tag whose version disagrees with
Cargo.toml, CITATION.cff, the citation snippets or the changelog fails
before anything is built or published. The Python package takes its version from Cargo.toml (maturin), so
it cannot drift on its own.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read(path: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return fh.read()


def changelog_section(text: str, version: str) -> str | None:
    """Body of the `## <version>` section (heading may carry a date)."""
    m = re.search(rf"^## v?{re.escape(version)}\b[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="release tag, e.g. v0.1.0")
    ap.add_argument("--notes", help="write the changelog section to this file")
    a = ap.parse_args()

    if not re.fullmatch(r"v\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?", a.tag):
        sys.exit(f"tag {a.tag!r} is not v<major>.<minor>.<patch>[-pre]")
    version = a.tag[1:]
    errors = []

    cargo = read("Cargo.toml")
    ws = re.search(r'^\[workspace\.package\][^\[]*?^version = "([^"]+)"', cargo, re.M | re.S)
    if not ws or ws.group(1) != version:
        errors.append(f"Cargo.toml [workspace.package] version is {ws and ws.group(1)}")
    dep = re.search(r'^radveil-core = \{ version = "([^"]+)"', cargo, re.M)
    if not dep or dep.group(1) != version:
        errors.append(f"Cargo.toml workspace dependency radveil-core is {dep and dep.group(1)}")

    cff = re.search(r"^version: (\S+)", read("CITATION.cff"), re.M)
    if not cff or cff.group(1).strip("\"'") != version:
        errors.append(f"CITATION.cff version is {cff and cff.group(1)}")

    # The BibTeX snippets people copy into papers.
    for path in ("README.md", "site/index.html"):
        for found in re.findall(r"version\s*=\s*\{([^}]+)\}", read(path)):
            if found != version:
                errors.append(f"{path} citation version is {found}")

    notes = changelog_section(read("CHANGELOG.md"), version)
    if not notes:
        errors.append(f"CHANGELOG.md has no '## {version}' section (rename 'Unreleased')")

    if errors:
        sys.exit("release check failed for " + a.tag + ":\n  " + "\n  ".join(errors))
    if a.notes:
        with open(a.notes, "w", encoding="utf-8") as fh:
            fh.write(notes + "\n")
    print(f"{a.tag}: versions consistent, changelog section found")


if __name__ == "__main__":
    main()
