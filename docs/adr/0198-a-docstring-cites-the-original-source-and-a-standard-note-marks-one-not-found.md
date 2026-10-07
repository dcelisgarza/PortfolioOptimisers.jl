---
status: accepted
---

# A docstring cites the original source, and a standard note marks one that was not found

## Context

`mean` for `BodnarOkhrinParolya` divided the coefficient ``\beta`` by the quadratic form of the
sample mean. Equation 7 of `bodnar2019` divides by the quadratic form of the target. Equation 3.45
of `cajas2025` restates the paper with the wrong quotient. The method and its test copied the
restatement, and the docstring sweep of
[#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) checked the code against
that same restatement, so the check passed. `391eee1a52` fixed it after a check against the paper.

A check against a work that restates a formulation cannot find an error that the work carries. The
reference set of #404 told a session to prefer the book for a formulation the code builds, and that
advice is the path by which the defect came in. Many docstrings cite only a book, a survey or a
software manual. [#1547](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1547) lists
each of them, one audit ticket per child map of #404.

Some originals will not be found. On 2026-10-07 the maintainer asked that a docstring whose
original source cannot be found carries a standard note in the relevant entry.
[#1533](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1533) settled the rule and the
form of the note.

## Decision

**Three rules, which `.github/instructions/julia-docstrings.instructions.md` § *The original
source* states.**

1. A formulation is checked against its **original source**, the work that first states the
   formulation that the code builds. A work that restates it is a **secondary source** of it.
   Secondary is a relation between a work and one formulation, not a kind of work: a book is the
   original source of a formulation that it states first.
2. A docstring cites the original source. It cites a secondary source too only where that source
   adds something, and the sentence says what it adds.
3. When the original cannot be found, the docstring carries the **standard note**.

**The form of the note, which § *The `# References` Section* states.**

- The note is one key, `ref_dict[:no_original_source]`, with the text "The original source of this
  formulation was not found. The formulation follows this work and is not checked against the
  original." Its wording cannot drift, and a search for the key lists every open formulation.
- The note closes the `# References` bullet of the secondary source that the docstring follows,
  after a mandatory locator: `- $(ref_dict[:cajas2025]) Equation 3.45. $(ref_dict[:no_original_source])`.
  A reader looks in that bullet for the source of a formula, and a later search starts from the
  locator.
- `test/test_26_docs.jl` checks the form alone: the note closes a `# References` bullet after a
  locator, and it appears nowhere else. No ratchet counts the notes, because a note is the honest
  result of a search.
- `code_health/sweep_triage.jl` writes the rule into condition 2 of each sweep sub-issue it files.

## Considered and rejected

- **The note in `val_dict`.** That table holds the lines of a `# Validation` section, and its name
  would mislead a reader. The note belongs to a `# References` bullet, and `ref_dict` holds those.
  The cost is one key of `ref_dict` that names no entry of `docs/src/References.bib`, which the
  check that each key names an entry skips by name.
- **The note in the sentence that states the formulation.** It is more precise for a docstring that
  follows a secondary source for one formula and an original for another. The mandatory locator
  gives the same precision in the bullet, and a census then reads one form instead of every
  sentence.
- **A gate that refuses the note after a key that is not secondary.** A fixed set of secondary keys,
  such as every `@book`, cannot be right, because a work is secondary for one formulation and
  original for another. Such a gate would refuse an honest note on an article that restates an
  older one, and it would accept a note on a book that first states the formulation.

## Consequences

The key lands with its first user, an audit ticket of #1547, because a `ref_dict` entry with no
user fails `test/test_26_docs.jl`. Until then the check of the form is vacuous on the source
tree, so it is pinned on a synthetic text.

A `# References` bullet with no note now states that every formulation the docstring takes from
that work was checked against its original source. A sweep sub-issue filed from now on carries the
rule in its condition 2. The sub-issues filed before it carry the older line, and #404 § *The
original source* states the rule for them.
