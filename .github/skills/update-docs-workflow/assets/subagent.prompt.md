Update all affected project documentation for the current repository based on the provided change summary and affected document scope.

Your job is to decide which documents should change, update them in the right order, and stop immediately if you discover a blocker that the main session must resolve.

## Inputs You Should Expect
- A summary of the implementation or project change
- The affected document scope, if already known
- Optionally, hints about files that probably changed or should be checked
- Prefer structured input with `change_summary`, `affected_scope`, `files_touched`, `known_doc_targets`, and optional `documentation_expectations`

## Core Rules
- Do not treat current-state snapshot documents as append-only logs.
- If the repository has a document like `docs/CONTEXT.md`, treat it as the current-state snapshot. Rewrite it in place only when the best concise description of the present state has changed.
- If the repository has a document like `docs/DEV.md`, treat it as append-only and only for durable historical context: intent, decisions, verification, and rationale worth preserving. Append to it once per resolved problem or meaningful milestone, not once per edit or per conversation turn.
- If the repository has a document like `refs/REF.md`, treat it as curated reference knowledge, not project progress tracking. Do not update it unless the change explicitly affects durable, reusable reference knowledge.
- If the repository has workflow guidance like `.github/copilot-instructions.md`, treat it as high-impact and comparatively stable. Only update it when repository-wide guidance, document ownership, or repeatable workflow expectations materially changed.
- `README.md` means any affected README in scope, including the project-root README and directory-level `*/README.md` files. Root README files usually explain project purpose, setup, usage, and entry points. Directory-level README files usually explain local conventions, structure, ownership, or usage within that subtree. Update a README only when those responsibilities materially changed.
- Default to no documentation change if the existing docs are still materially accurate.
- Prefer compressing or rewriting stale snapshot content over appending incremental detail.
- Do not add low-signal implementation trivia to any document.
- Unless the user explicitly overrides this, this run owns updates to current-state snapshot docs, append-only development logs, workflow instructions, and affected README files, but workflow guidance docs should face the highest bar for change.

## Document Ownership Heuristics
- Lowest-level affected docs first: local `*/README.md`, folder conventions, or narrowly scoped reference docs
- Then project reference and workflow docs: reference indexes, workflow instructions, prompt or agent docs if affected
- Then project-level snapshot and history: current-state snapshot docs and append-only development logs
- In short: update from the most local impacted document upward toward the highest-level project summaries

## Required Workflow
1. Read the change summary and determine the candidate documentation set.
2. Inspect the affected files or directories as needed to gather the missing facts required for documentation updates.
3. Detect which files in the current repository play the role of snapshot docs, append-only development logs, reference indexes, workflow instructions, and README files.
4. Decide which documents actually need changes and which should remain untouched.
5. Apply documentation updates from the bottom up:
   - Start with the most specific affected docs
   - End with top-level snapshot or historical docs
6. If you discover a blocker, inconsistency, missing fact, or conflicting state that prevents correct documentation updates, stop immediately.
7. Return the problem clearly to the main session instead of guessing.
8. If no blocker exists, finish all needed documentation updates before returning.

## Blocker Policy
Return early without partial guesswork if:
- The requested documentation outcome conflicts with the actual repository state
- The source change is too ambiguous to document accurately
- Required facts are missing and cannot be verified from the workspace
- There are conflicting documents and you cannot determine the correct owner without a user decision

## Output Format
If blocked, return:
- `Status: blocked`
- The exact blocker
- Which files were inspected
- What the main session needs to resolve before rerunning

If successful, return:
- `Status: updated`
- Modified files
- One-line summary of what changed in each file
- Files considered but intentionally left unchanged
