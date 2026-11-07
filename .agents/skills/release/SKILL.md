---
name: release
description: "Prepare, approve and verify an fse release: CHANGELOG section, frozen persistence names, green master CI, maintainer approval, tag, the `release` environment, then PyPI and the GitHub release. Use for every release of fse, including pre-releases. Never tags or publishes without the maintainer's explicit approval."
---

# Release fse

A pushed `v*` tag on `master` runs the release job in `.github/workflows/ci.yml`. The job publishes to PyPI. The job `github-release` then creates the GitHub release. Both are one release.

## Hard rules

- **Never create a tag, a release or a publication without the maintainer's explicit approval for this version.** Approval of an earlier version, of this checklist or of a pull request is not approval to tag.
- Never approve the `release` environment yourself, never push a tag to test the workflow, never create the GitHub release by hand for a tagged version (the job does it; a manual one blocks the job).
- A tag cannot be undone: PyPI never accepts a file name twice.

## Checklist

1. **Version and scope.** Name the version `X.Y.Z` (pre-release: `X.Y.ZaN`, `bN` or `rcN`) and the commit on `master` that is released. Version numbers come from the tag through setuptools-scm.
2. **CHANGELOG.md.** Rename `## [Unreleased]` content into `## [X.Y.Z] - YYYY-MM-DD` and leave an empty `## [Unreleased]` above it. The section is the text of the GitHub release, so it must read well on its own. The job refuses a tag without that exact heading, an empty section or a duplicate.
3. **Rehearse the notes** without a tag: `python3 tools/release_notes.py --tag vX.Y.Z --output <scratch file>`. Read the output.
4. **Persistence names.** From the first release of 2.0 on, persistence names are frozen (docs/architecture/persistence.md, "Persistence names replace class paths"). Check that no name released before was renamed or reused, and that new names are intended to freeze.
5. **Dependency lower bounds** in `pyproject.toml` match the minimum job pins in `.github/workflows/ci.yml`.
6. **Green master CI on the exact commit**: a full `push` run on `master` (not a skill-only run) with `wheels (ubuntu-latest, manylinux)` successful. The release job checks this again and refuses otherwise.
7. **Maintainer approval.** Ask for the explicit go for `vX.Y.Z` at the named commit. Stop here until it is given.
8. **Tag.** Only after the go: create the annotated tag `vX.Y.Z` on that commit on `master` and push it.
9. **Approve the environment.** The job waits in the `release` environment. The maintainer approves it. Watch the run in a background monitor, never in the foreground.
10. **Verify afterwards**, and report each result:
    - PyPI: `https://pypi.org/project/fse/X.Y.Z/` lists the sdist and the wheels for every platform.
    - GitHub release: `gh release view vX.Y.Z --json name,tagName,isPrerelease,body,assets`. Name is `fse X.Y.Z`, the body is the CHANGELOG section, the assets are the sdist and the wheels, and a pre-release is marked as one. `gh api repos/oborchers/Fast_Sentence_Embeddings/releases/latest --jq .tag_name` names `vX.Y.Z` for a final release and still the older one for a pre-release.
    - A clean install of `fse==X.Y.Z` imports.

## If a job fails

- **Before the PyPI upload** (CHANGELOG, tag or file names): nothing is published. Fix on `master` through a pull request. A tag already pushed points at the old commit; ask the maintainer how to proceed, never move or delete a tag yourself.
- **After the PyPI upload, in `github-release`**: PyPI is final. Re-run the failed job. Only if that cannot work, and only with approval, create the release by hand with the flags of the job: `gh release create vX.Y.Z --verify-tag --title "fse X.Y.Z" --notes-file <notes> dist/*`, plus `--latest` for a final version or `--prerelease --latest=false` for a pre-release. The notes come from `tools/release_notes.py`, the files from the artifacts of the run.
