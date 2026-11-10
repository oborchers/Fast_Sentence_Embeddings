## Change and reason

Describe the problem and resulting behavior. Link the relevant issue with its title.

## Evidence

- Base commit:
- Head commit:
- Exact commands and raw result links:
- Data sources, revisions and seeds:
- CPU, instruction set, OS, Python, NumPy, compiler and relevant package versions:
- Platforms checked (x86 AVX2, ARM NEON, Linux, Windows), and untested platforms:
- Claim classes used (correctness, descriptive, regression, optimization, publication) and Linux comparison status (machine and results, or none run):
- Factual claims and their sources (including documentation changes):

For performance or hot-path changes, report paired alternating rounds, median and range, load, and an A/A control where the claim class requires them.
Include the kernel and end-to-end throughput comparisons that the chosen claim class and protocol require.
A Linux comparison is required only for class publication or when the maintainer has explicitly approved a Linux machine for the issue. Report its actual status, and state explicitly when none was run.
Attach CPU, allocation-memory and process-tree resident set size (RSS) profiles.
Otherwise state why these do not apply.

## Checks

- [ ] `make check` passes. Exact command and output are linked above.
- [ ] Relevant network and sentence-transformers checks pass, or omissions are explained.
- [ ] Conformance and bit identity are unchanged.
- [ ] No new ratchet entries or suppressions.
- [ ] User-visible changes have a `changelog.d/DR-<nnn>.<type>.md` fragment.
- [ ] Measurements cover the target environment. Limits and skipped checks are explicit.
- [ ] AI tools and model versions are disclosed below, or no AI assistance was used.
- [ ] I agree that my contribution is licensed under the MIT license as part of fse 2.0.
- [ ] Third-party code and data, if any, have identified sources and licenses.

## AI assistance

Name tools and model versions, their role, and what you verified yourself.
