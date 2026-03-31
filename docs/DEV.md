# Development Log

> Append-only. Records intent, decisions, and verification — what git log doesn't capture.
> Non-essential reading. Consult only when you need historical context.

---

## [2026-03-31] chore: initialize project infrastructure

### Scope
- `.github/copilot-instructions.md` — created (replaces old deleted version)
- `docs/` — created (CONTEXT.md, DEV.md, plans/README.md)
- `refs/README.md` — created

### Motivation
Previous project documentation was loosely organized (`CONTEXT.md` and `plan.md` at repo root, later moved to `NO-USE/`). Old `.github/` directory with skills files was deleted. This initialization establishes a clean, structured documentation system for AI-assisted development workflow.

### Decisions
- Used `docs/` for project documentation, `.github/` for AI configuration only
- Old `NO-USE/CONTEXT.md` content migrated and restructured into `docs/CONTEXT.md`
- Old `NO-USE/plan.md` (hover training plan) preserved for reference, to be migrated to `docs/plans/` when work begins
- Conventions stored as README.md in target directories (discoverable by any tool or human)
- Development workflow follows TDD-oriented checklist
- Plans and references are generated on-demand, not upfront
- Old `.github/skills/` not restored — those were tightly coupled to the previous setup

### Verification
- [x] All scaffold files created
- [x] copilot-instructions.md reflects actual project state
- [x] CONTEXT.md contains real architecture and pitfall data from codebase exploration
