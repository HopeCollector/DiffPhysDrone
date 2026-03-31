# Reference Documentation

## Gathering Raw Docs

Store version-matched documentation in `<tool-name>/` subdirectories:

- Fetch from official git repos or documentation sites (use git sparse-checkout or targeted downloads)
- Must match the **exact version** used in this project
- Prefer markdown or plain text

## Curating REF.md

Create `REF.md` in this directory when you start accumulating references.

REF.md is a **project-specific index** of useful knowledge extracted from raw docs. It is NOT a dump of full documentation.

Each entry must include:

1. **Section header**: Tool/library name and the specific capability
2. **One-line summary**: What it does and when you'd use it
3. **Minimal example**: Working code snippet
4. **Source link**: Relative path to the full doc within this folder
5. **Version and verification date**: For freshness tracking

Keep entries concise. Link to full documentation — never copy large sections into REF.md.
