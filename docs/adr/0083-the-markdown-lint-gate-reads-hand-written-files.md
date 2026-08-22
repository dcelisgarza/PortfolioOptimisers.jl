---
status: accepted
---

# The markdown lint gate reads hand-written files, and a duplicate heading is judged among siblings

## Context

`pre-commit run -a` could not pass on `dev`. Every hook passed except `markdownlint-fix`, which
reported **166 errors over 20 files**. The failure was not new: the two oldest offending files were
last written at `c238c76408`, the commit `main` sat at. Issue
[#413](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/413) recorded it, and
[#409](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/409) found it, because a clean
`pre-commit run -a` is that ticket's definition of done.

Three groups produced the 166 errors, and two of them could not be fixed at the file the error
named.

**`MD024/no-duplicate-heading` contradicted this repository's own rule for an ADR.** `CLAUDE.md`
states that an ADR whose decision reached `main` is amended, never rewritten: append a
`## Amendment (YYYY-MM-DD)` section. An amendment that restates the decision it changes therefore
repeats the heading `Decision`, `Verification` or `Rejected alternatives`, and `MD024` forbids
exactly that. Ten ADRs carried the repeat. No edit to an ADR could satisfy both rules.

**A generated page cannot be corrected at its own path.** `CLAUDE.md` states that
`docs/src/examples/**`, `docs/src/user_guide/**`, `docs/src/capability_catalogue.md` and
`docs/src/api/*_TypeHierarchy.md` are written by CI from their `.jl` sources and are overwritten on
every build. Three of them carried an error.

**`research/` holds working notes.** Seven files wrote their numbered sections as `#` headings and
fired `MD025/single-title`, `MD001/heading-increment` and `MD036/no-emphasis-as-heading`. They are
research documents, not repository documentation, and their structure carries no reader contract.

The rest were nine hand-written defects in seven ADRs: seven code fences without a language, and
six table cells wider than their header, which `--fix` cannot repair.

## Decision

**The gate reads the markdown a person wrote, and it judges a duplicate heading among siblings.**

1. `.markdownlintignore` holds the scope. It lists the four generated paths and `research/`.
   markdownlint reads the file on its own, for a named path as well as for a glob, so a direct
   `markdownlint` run and the `pre-commit` hook see the same scope.
2. `.markdownlint.json` sets `"MD024": {"siblings_only": true}`. A heading now collides only with a
   heading under the same parent. Every ADR amendment is its own `## Amendment` parent, so
   `### Decision` under one amendment and `### Decision` under another are legal, and a genuine
   repeat inside one section still fails.
3. The nine hand-written defects are corrected. A fence takes `text`, or `julia-repl` where it
   holds a REPL transcript, and the three tables are padded to their widest cell.

`MD024` is scoped rather than turned off. `MD046` is turned off, because a Documenter admonition
body reads as an indented code block under every one of that rule's settings; `MD024` has a setting
that admits the amendment rule, so the rule keeps its remaining reach.

## Consequences

- `pre-commit run -a` passes, and `markdownlint-fix` rewrites nothing.
- A markdown defect in a generated page is fixed at the `.jl` source, or it is not fixed. The gate
  no longer reports it at a path no one may edit.
- `research/` is unlinted. A file moved out of `research/` enters the gate and must pass it.
- An ADR amendment may restate `Decision`, `Verification` and `Rejected alternatives`. This closes
  the conflict between `CLAUDE.md` and the lint configuration; neither rule bends further.
- `STANDARDS.md` routes markdown structure to `.markdownlint.json` and now names
  `.markdownlintignore` as the file that sets the scope.

## Verification

`markdownlint` over every tracked `.md` file reported 166 errors before the change and 0 after it.
`pre-commit run markdownlint-fix -a` passes and modifies no file. The seven ADRs whose text changed
differ from their committed content by padding and by a fence language alone.
