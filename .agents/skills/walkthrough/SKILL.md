---
name: walkthrough
description: Explain and demonstrate one fse implementation worktree or GitHub pull request, grounded in code and tests, one item at a time. Use when explicitly invoked with $walkthrough or directly asked to walk through fse work. Continue the active walkthrough when the user replies next.
---

# Walkthrough

Explain one bounded body of fse work one concise item at a time. Resolve the
source from context, prepare the complete agenda privately, and reveal only
the current item. Use German in a German context, otherwise match the user.

## Resolve and verify the source

1. Infer the current or named implementation worktree, or one specified open
   or merged GitHub pull request, from the request and recent conversation.
2. Read applicable `AGENTS.md` instructions. Inspect branch commits, status,
   and the complete diff against the verified base. For a PR, verify its base,
   exact head SHA, state, and merge commit through read-only GitHub queries.
3. Fetch `origin/master` when permitted. Confirm shipped work is reachable
   from that verified revision. Label open PRs, unmerged branches, and local
   changes as unshipped. If remote verification is unavailable, state that limit.
4. Ground claims in real code, tests, documentation, or observed behavior at
   the selected revision. PR descriptions and tickets alone prove no behavior.
5. Ask one focused scope question only if inspection cannot resolve the source.

Do not edit product code, repository documents, PRs, GitHub state, or settings.
Isolated test state and disposable demonstrations are permitted. Any setup or
build must respect current resource reservations and repository instructions.

## Select the checkout

Use a checkout that faithfully contains the source before presenting item one.
Record its path, branch, exact HEAD, and initial status. If it is dirty, record
which diff is part of the source and distinguish it from the committed revision.

- Reuse a matching linked implementation worktree. Preserve it, its agents,
  and its pre-existing services throughout the walkthrough.
- Use the primary checkout only for selected work already merged to `master`
  and present there. It is source-only for unmerged work.
- For committed unmerged work without a matching linked checkout, fetch the
  exact head and create a collision-free detached temporary worktree. Record
  its parent, path, and SHA. Do not create a walkthrough branch.
- If a demo depends on uncommitted primary-checkout changes, pause and ask the
  user to commit them or move them into a dedicated implementation worktree.
  Never copy, stash, reset, or clean their changes to manufacture a revision.

## Prepare the complete agenda privately

Freeze the full agenda before item one. Reveal its total count, but no future
item titles. Group by coherent behavior, contract, or operational change,
not by file, commit, or test. If new evidence changes the agenda materially,
explain why and rebuild it instead of silently changing the order or count.

For each item, decide whether a small API, CLI, or data demonstration helps.
fse is a CPU inference library for finished static embeddings, not a training
system. Use the current 2.0 API and code at the selected revision. The removed
1.x API belongs to branch `1.x`; do not present it as current behavior.

- Prefer tiny synthetic word vectors, sentences, and temporary artifacts.
  Derive examples from real signatures and tests in the selected checkout.
  A typical API demo uses `WordVectors.from_arrays`,
  `Encoder.from_word_vectors`, and `encode`; verify their current arguments.
- Use documented commands only when the selected change has a CLI entry.
  Do not invent a general fse CLI or carry over Cargo or Proqi TUI setup.
- For persistence, search, or validation, show one observable result, such as
  the output shape, a round trip, or the exact expected exception.
- Run examples against the exact checkout with its existing environment if
  available. If setup is needed and permitted, use uv and the Makefile.
  Do not download models, run benchmarks, or perform heavy builds merely
  to explain a change. Respect resource reservations even for setup.
- Read-only inspection and tiny offline demos need no Herdr or terminal UI.
  A live visual demo is useful only if the selected work actually adds visual
  behavior. Prepare its real state before describing the observed result.
- Distinguish tests inspected, tests executed, expected output, and observed
  output. If a demo is blocked, explain the limit rather than claiming it ran.

## Keep strict pace

Advance only when the user's entire reply is `next`, ignoring case, surrounding
whitespace, and terminal punctuation. Every other reply discusses the current
item. Answer it concisely and keep that item active. Do not offer back, skip,
or overview controls. Never show more than one agenda item per response.

Use roughly 100 to 250 words per item when useful, less for a simple change:

```text
2 / 5: <current item>

<what it means and why it exists>

<one optional tiny demo and its observable result>

Schreib next.
```

End each non-final item and discussion reply with `Schreib next.` in German
or `Say next.` in English. For a final item with a live demo still under
inspection, keep it available and end with `Letzter Punkt. Schreib next,
wenn du fertig bist.` (English: `Final item. Say next when you are finished.`).
That next ends inspection, not another agenda item. Otherwise end the final
item with `Walkthrough abgeschlossen.` or `Walkthrough complete.`

## Retain ownership and require cleanup approval

Track every temporary checkout, directory, process, and optional pane created.
Never inherit automatic deletion permission from another project's skill.
Completion, abandonment, and `next` do not authorize removal. Ask for explicit
approval of any cleanup action that requires it under local rules. Until then,
retain the resources and report their exact paths or IDs and ownership.

Immediately before an approved destructive action, resolve each target through
structured state and show its ID, name, and path where available. Verify that
it matches the recorded walkthrough-owned target. Act on one verified target
per call, then list again to confirm only that target disappeared. For a
walkthrough-created detached worktree, also verify its recorded HEAD and empty
tracked and untracked status. Remove without force only after approval.

Preserve every reused implementation worktree, agent, pane, and pre-existing
service. If ownership, identity, revision, or cleanliness differs, retain the
resource and report why. Never force cleanup. At the end, report created
resources that remain, approved cleanup results, and the reused checkout's
path, branch, and status when applicable.
