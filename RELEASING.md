# Releasing

Releases are cut by pushing a version tag. [`release.yml`](.github/workflows/release.yml)
then builds and tests everything and, only if all of it passes, publishes:

| Channel | What | How |
|---|---|---|
| PyPI | `radveil` wheels (abi3, Python ≥ 3.9) for Linux x86_64/aarch64 (manylinux), macOS x86_64/arm64, Windows x86_64, plus the sdist | trusted publishing |
| crates.io | `radveil-core` (library), then `radveil-cli` (`cargo install radveil-cli`) | trusted publishing |
| GitHub release | `radveil-vX.Y.Z-<target>.tar.gz` / `.zip` per target: `bin/radveil` CLI, `lib/` shared + static C library, `include/radveil.h`, license files; all Python artifacts; `SHA256SUMS`; build-provenance attestations | `gh release create` |

`radveil-py` and `radveil-capi` are not published as crates (`publish = false`):
they ship as wheels and as the prebuilt C library. The C API is plain C, so the
same archives serve C++ projects (`#include "radveil.h"`, link `-lradveil`).

## Per release

1. Update the version in:
   - `Cargo.toml` (`[workspace.package] version` and the `radveil-core`
     workspace dependency). The Python package takes its version from here.
   - `CITATION.cff` (`version`, `date-released`).
   - The BibTeX and methods-sentence snippets in `README.md` and `site/index.html`.
2. In `CHANGELOG.md`, rename `## Unreleased` to `## X.Y.Z — YYYY-MM-DD`. The
   section becomes the GitHub release notes.
3. Check locally: `python tools/release_check.py vX.Y.Z` (versions agree,
   changelog section exists) and `python tools/third_party_licenses.py --check`.
4. Optional dry run: Actions → Release → Run workflow. It builds and tests every
   artifact but publishes nothing.
5. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "RadVeil X.Y.Z" && git push origin vX.Y.Z
   ```

Tags with a suffix (`v0.4.0-rc.1`) are published as pre-releases on GitHub;
PyPI and crates.io treat them as pre-releases too.

If a job fails after something was published, fix the cause and use
"Re-run failed jobs". Uploads already on PyPI are skipped, and so are crate
versions already on crates.io. Never move or reuse a tag once anything from
it has been published; release a new patch version instead.

## One-time setup (new repository)

- **GitHub environments** `pypi` and `crates-io` (Settings → Environments).
  Adding yourself as a required reviewer makes every publish wait for a click.
- **PyPI**: add a pending trusted publisher for project `radveil`: owner
  `rAum`, repository `radveil`, workflow `release.yml`, environment `pypi`.
- **crates.io**: trusted publishing can only be configured for a crate that
  already exists. For the first release, create an API token scoped to
  `publish-new`/`publish-update`, store it as the repository secret
  `CARGO_REGISTRY_TOKEN`, and let the workflow publish. Then, for both
  `radveil-core` and `radveil-cli`, add the trusted publisher on crates.io
  (repository `rAum/radveil`, workflow `release.yml`, environment
  `crates-io`) and delete the secret and the token.
- **GitHub Pages**: Settings → Pages → Source: GitHub Actions (`pages.yml`
  deploys `site/` on pushes to `main`).
- **Private vulnerability reporting**: Settings → Security → enable it
  (`CONTRIBUTING.md` points reporters there).
- Optional: protect `v*` tags (Settings → Rules) so that only maintainers can
  create them.

Publish only artifacts built by the workflow. A wheel or binary built on a
personal machine embeds that machine's absolute source paths (in panic
messages and the wheel's SBOM).

## Verifying a download

```bash
sha256sum -c SHA256SUMS --ignore-missing
gh attestation verify radveil-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz --repo rAum/radveil
```

## Maintenance

Actions are pinned to commit SHAs. Dependabot opens weekly PRs for them
and for Cargo dependencies. A dependency update changes the bundled
notices: run `python tools/third_party_licenses.py` on the PR and commit the
regenerated `THIRD_PARTY_LICENSES.txt`, or CI fails.
