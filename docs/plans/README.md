# Feature Plans

Create plan files as `<feature-name>.plan.md` in this directory.

## Rules

- Divide work into **numbered phases**
- Each phase contains only:
  - **Goal** — What is observably different when this phase is done
  - **Done When** — Verifiable acceptance criteria (checkboxes)
  - **Depends On** — Prerequisite phases, if any
  - **Risks** — Unknowns or areas needing spike/research
- **Never prescribe implementation details** — those depend on the codebase's state at development time
- Phases should be completable in a single development session

## Phase Template

## Phase N: \<Name\>

**Goal**: \<What changes — observable by user or developer\>

**Done When**:
- [ ] \<Specific, testable criterion\>
- [ ] \<Another criterion\>

**Depends On**: Phase X | —

**Risks**: \<None | describe unknowns\>
