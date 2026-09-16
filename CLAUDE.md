# mlsys-notebook

A personal notebook of ML systems concepts. One folder per concept, each explaining a problem and how the concept solves it.

## Structure

```
<concept-name>/
  README.md        # the explainer — required, the only thing that must exist
  images/          # diagrams referenced by README.md
  code/            # runnable snippets referenced by README.md
```

- Folder names are kebab-case and named after the concept (`kv-cache`, `tensor-parallelism`, `speculative-decoding`), not after a paper or product.
- `README.md` is the entry point so the folder renders on GitHub when browsed.
- Supporting files exist only to be referenced from `README.md`. An artifact no prose points to is dead weight — delete it or link it.

## Writing a concept note

Use the `note` skill (`.claude/skills/note/SKILL.md`) — it holds the note structure, sourcing rules, and style bar.

## Conventions

- Reference sibling files with relative paths (`./images/foo.png`, `./code/bar.py`), never absolute paths or raw GitHub URLs.
- Link between concepts with relative folder links (`../kv-cache/README.md`) — the notebook's value is in the cross-references.
- Code in `code/` is illustrative, not a library. Keep it self-contained and runnable on its own; no shared package, no imports across concept folders.

## Gotchas

- This is prose-first, not a codebase. There is no build, no test suite, and no dependency file — don't add one for a single snippet.
- Do not create a concept folder without content. An empty folder with a stub `README.md` is worse than no folder.
