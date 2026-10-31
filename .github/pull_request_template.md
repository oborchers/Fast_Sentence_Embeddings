## Change and reason

Describe the problem and resulting behavior. Link the relevant issue with its title.

## Evidence

- Base commit:
- Head commit:
- Exact commands and raw result links:
- Data sources, revisions and seeds:
- CPU, instruction set, OS, Python, NumPy, compiler and relevant package versions:
- Platforms checked (x86 AVX2, ARM NEON, Linux, Windows), and untested platforms:
- Factual claims and their sources (including documentation changes):

For performance or hot-path changes, include paired alternating rounds, median and range,
load, an A/A control, required kernel and end-to-end throughput comparisons, and CPU,
allocation-memory and process-tree RSS profiles. Otherwise state why these do not apply.

## Checks

- [ ] `make check` passes. Exact command and output are linked above.
- [ ] Relevant network and sentence-transformers checks pass, or omissions are explained.
- [ ] Conformance and bit identity are unchanged.
- [ ] No new ratchet entries or suppressions.
- [ ] User-visible changes have a `CHANGELOG.md` entry.
- [ ] Measurements cover the target environment. Limits and skipped checks are explicit.
- [ ] AI tools and model versions are disclosed below, or no AI assistance was used.
- [ ] I agree that my contribution is licensed under the MIT license as part of fse 2.0.
- [ ] Third-party code and data, if any, have identified sources and licenses.

## AI assistance

Name tools and model versions, their role, and what you verified yourself.
