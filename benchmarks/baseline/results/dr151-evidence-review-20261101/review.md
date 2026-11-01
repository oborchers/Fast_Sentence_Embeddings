No findings at committed `267cdd3`.

1. **Restore safety: no findings.** The helper validates the relative path, unique manifest entry and full pinned object ID. It checks size and SHA256 before writing. Exclusive creation prevents overwrite.

2. **Preservation tests: no findings.** Source assertions compare matrix rates, A/A values, runtime configurations and RSS. They also compare catalog values and exact original bytes for both retained Pod3 summaries. Missing archive objects trigger an explicit skip.

3. **Baselines and references: no findings.** All eight original 1.1 baselines remain byte-identical. All 465 manifest entries match the pinned archive, totaling 67,727,161 bytes. No undocumented dependency on deleted raw files was found.

4. **Summaries and claims: no findings.** Retained fields match across 54 matrix cells, 30 Pod3 panel cells and both selected Pod2 cells. Configuration indices remain storage references. Missing fields remain null. Score, protocol, normalization and noise qualifications remain intact. Pod4 remains rejected with `eligible: false` and global A/A ratio `1.1506593476578448`.

Done: source-only review complete. No tests, builds, project/model imports, downloads or model workloads ran.  
Open: nothing.  
Needs from you: nothing.