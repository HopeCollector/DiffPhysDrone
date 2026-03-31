---
name: update-docs-workflow
description: 'Run a documentation update workflow after a resolved change or meaningful milestone. Use when documentation should be reviewed in a fresh subagent, with a structured change summary, affected scope, blocker-first behavior, and bottom-up document updates.'
argument-hint: 'Describe the resolved change, affected scope, and any documentation expectations.'
user-invocable: true
---

# Update Docs Workflow

Use this skill when documentation should be updated only after a problem is solved or a meaningful milestone is complete.

## When to Use
- A feature, fix, or infrastructure change has been completed and project documentation may now be stale
- You want documentation updates to run in a fresh, context-isolated subagent instead of being mixed into the main implementation thread
- You need blocker-first behavior: if the documentation state is ambiguous, stop and hand the problem back to the main conversation
- You want lower-level docs updated before higher-level snapshot or history docs

## Do Not Use
- For intermediate edits while a problem is still being worked through
- When the user explicitly wants direct manual edits in the main conversation
- When the task is only to record reusable technical knowledge in `refs/REF.md`

## Resources
- [Subagent prompt](./assets/subagent.prompt.md)

## Procedure
1. Confirm that the underlying problem is solved or that a meaningful milestone is complete.
2. Gather a concise change summary, affected scope, touched files, likely doc targets, and any documentation-specific constraints.
3. Start a fresh, context-isolated subagent.
4. Inject the contents of [Subagent prompt](./assets/subagent.prompt.md) together with a structured payload like:

```text
change_summary:
<concise summary of what changed>

affected_scope:
<directories, files, or document layers likely affected>

files_touched:
- <file 1>
- <file 2>

known_doc_targets:
- <doc that likely needs review>
- <doc that likely does not need review>

documentation_expectations:
- <optional constraints, priorities, or known user preferences>
```

5. Let the subagent inspect repository state, decide what to update, and either finish all documentation updates or return immediately with a blocker.
6. If the subagent reports a blocker, resolve it in the main conversation and restart a brand-new subagent run instead of trying to continue the previous documentation thread.
7. If the subagent succeeds, review the returned modification summary and only then continue with the main conversation.

## Quality Bar
- Documentation changes should be high-signal and role-consistent.
- Current-state snapshot docs should be rewritten in place, not treated as changelogs.
- Append-only dev logs should receive at most one entry per resolved problem or meaningful milestone.
- Workflow guidance docs such as `.github/copilot-instructions.md` should be treated as high-impact and comparatively stable. Only change them when repository-wide guidance or document ownership rules materially changed.
- README updates should follow document ownership and only happen when their responsibilities materially changed.
- Reusable knowledge notes should stay separate from progress tracking.

## Completion Check
- A fresh subagent was used
- The subagent received a structured payload
- Documentation was updated bottom-up or a blocker was returned immediately
- The main conversation received a concise summary of modified and intentionally untouched docs
