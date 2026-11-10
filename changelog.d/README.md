# Changelog fragments

Add one `DR-<nnn>.<type>.md` file for each user-visible change. Use your ticket ID.
Write English text that explains the resulting behavior. Do not edit CHANGELOG.md.
If you mention a ticket in the text, include its title. Do not start with a bullet.
Multiple paragraphs and code blocks are allowed. Indent continuation paragraphs normally;
towncrier adds the indentation of the outer bullet.

| Type | Meaning |
| :--- | :--- |
| added | New behavior |
| changed | Changed behavior |
| deprecated | Behavior scheduled for removal |
| removed | Removed behavior |
| fixed | Corrected behavior |
| security | Security correction |

The PR gate requires a new nonempty fragment for changes under `src/`, `setup.py`,
`MANIFEST.in`, and changes to these pyproject.toml tables: `build-system`, `project`,
`tool.setuptools`, `tool.setuptools_scm` (including their subtables).
Pure tests, documentation, benchmarks, tools, CI, development groups, other tool
configuration and lockfile-only changes do not trigger the automatic requirement.
This is the gate's boundary. Any user-visible change still needs a fragment,
even outside those paths. Editing CHANGELOG.md does not bypass the gate.

To inspect fragments without writing files:

```sh
uv run --group lint towncrier build --draft --version 2.1.0
```

Only release preparation builds the real changelog and consumes fragments.
Follow [.agents/skills/release/SKILL.md](../.agents/skills/release/SKILL.md).
README.md and template.md.jinja are metadata, not fragments. Leave no other files
or directories here after the release build. Publishing rejects them.
