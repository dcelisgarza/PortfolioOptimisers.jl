---
applyTo: 'examples/**/*.jl, user_guide/*.jl, docs/src/**/*.md, README.md, docs/capability_catalogue.jl'
---

# User-Facing Prose Guidelines for PortfolioOptimisers.jl

This file governs the **prose a user reads as a page**. Four corpora carry it.

- **The Literate sources** under `examples/` and `user_guide/`, from which the docs build renders
  the examples and the user guide. Their prose is the text of every `#= … =#` block, every comment
  line that Literate renders as Markdown, and the `Description` line of a ` ```@meta ` block.
- **The hand-written Markdown pages** under `docs/src/`, outside `docs/src/contribute/`, which is
  written for a contributor. Their prose is the text of the page.
- **`README.md`**, the repository front page. It is the one user-facing text at the root; every
  other Markdown file there is written for a contributor.
- **`docs/capability_catalogue.jl`**, from which the docs build renders the Capability Catalogue.
  Its prose is the text of every `Prose`, every `Note`, every `Cap` label, every `Section` title
  and every `Group` head that is a string. A `Cap` with no label takes its sentence from the type's
  docstring, which [`julia-docstrings.instructions.md`](julia-docstrings.instructions.md) owns.

A line of Julia, a fenced code block, a LaTeX expression, an `@ref` link and a `#src` line are
outside this file. So is a docstring, which reaches a page through `@docs` and which the docstring
standard owns. *What the rule does not read* below states the rest.

On a Literate page a reader runs the cells in order, and the prose between two cells says what the
next cell does, or what the last output shows.

---

## The prose passes `/unslop`

The `unslop` skill states the patterns that mark generated text, and every rule of the skill holds
on the prose of these files. **The skill is the Authority for the patterns.** This file cites it by
name, copies none of its rules, and names a rule by the number the skill gives it.

Rules 20 and 22 describe a chat reply. They never arise on a page. Every other rule applies as
written.

**The prose carries no dash of either kind.** Rule 13 removes every em dash. An en dash goes with
it: a compound name takes a hyphen, `Black-Litterman`, and a numeric range takes a hyphen or the
word `to`.

**How to apply it.** Invoke `/unslop` on the file. When the skill cannot be invoked, read its rules
and apply them by hand. Rewrite the prose, keep the meaning, and end with the skill's self-audit,
"What makes this obviously AI generated?". A pass changes prose alone: a line of Julia, a `#src`
line, an `@ref` link, an `@docs` block, an admonition's indentation and a fenced block stay as they
are, and the outputs of a page do not move.

---

## The page speaks to the reader

- **`you` names what the reader does, chooses, or reads off an output.** "Reach for `MeanRisk`
  when you want the trade-off between return and risk." "The table you get back holds one row per
  fold."
- **`we` names what the page's cells do.** "We fit the prior on the first training window."
- **The impersonal third person names what the library does.** "`prices_to_returns` carries every
  gap into the returns."

**Three rules of this file read a page a reader runs, and a text with no cells is outside them.**
They are `we` for the page's cells, *A paragraph carries one job* below, and *The page's furniture*
below. `you` for the reader and the impersonal third person for the library hold on every text this
file governs, and so does every other rule here.

---

## A page names a type by its identifier and a concept in plain words

- **A type is its identifier**, in a code span, with an `@ref` where the page first names it:
  [`CoveragePolicy`](@ref).
- **A concept is a plain phrase in lower case**, and the page defines it in the sentence where it
  first appears. Write "the assets of the training window with enough observations, the coverage
  universe", then "the coverage universe" from there on. The capitalised forms that `CONTEXT.md`
  defines, such as `Coverage Universe`, `Panel Field` and `Online Scheme`, are for a contributor
  reading `CONTEXT.md`. A reader of a page has no glossary.
- **A word for the mechanism of the code never appears.** "seam", "carrier", "read-out", "host"
  for a type that holds another, "to the bit", "refused by name", and the section sign `§`. Say what
  happens instead: "the optimiser hands its prior the new rows", not "the optimiser is the host of
  the seam".

---

## A check is a number the reader reads, never a verdict

A page can run a comparison and print its result. The prose says what the cell computes and what
the printed number means for the reader.

The prose never states the outcome as proved, and it never borrows the words of a test: "to the
bit", "by construction", "measured honestly", "the identity", "agrees".

Write "The cell prints the largest difference between the online weights and the batch weights over
every fold. It is zero, so the switch changes no weight." Do not write "The run agrees with the
batch run to the bit, by construction."

---

## A paragraph carries one job

This section reads a page a reader runs. A paragraph between two cells says what the next cell
does, or what the last output shows. It restates neither the code nor the number the cell prints.

A page carries no word band. A page is long because it covers more ground, and the paragraph rule
is what keeps it from being long because it repeats itself.

---

## The page's furniture

This section reads a page a reader runs. Three elements are permitted on any such page, expected on
a long one with many sections, and required on none.

- The `!!! tip "When to reach for this"` admonition under the H1, which says when a reader reaches
  for the estimator and what to reach for instead.
- A numbered outline of the page's sections under the H1.
- A closing `## What to take away` section. Rule 25 binds it: it states the facts the page
  measured, never a generic close.

A short page carries none of them and is complete without.

---

## What the rule does not read

**Derived text.** The rule reads written prose. A text that a script derives, or that another
census holds to a shape, is outside it, because two gates over one line disagree sooner or later.
Two such texts exist, and a page's path tells them apart, as
[`test/test_64_docs_page_metadata_census.jl`](../../test/test_64_docs_page_metadata_census.jl)
already does:

- **The H1 of a mirror page** under `docs/src/public_api/` or `docs/src/private_api/`, whose shape
  ADR 0128 fixes and which ends in `: public API` or `: private API`.
- **The `Description` line of a mirror page**, which `docs/page_metadata.jl` derives from the names
  the page hosts, and which
  [`test/test_64_docs_page_metadata_census.jl`](../../test/test_64_docs_page_metadata_census.jl)
  fails when it drifts from that derivation.

The `Description` line of a Literate page and of a hand-written page is written prose, and the rule
reads it.

**Generated pages.** `docs/src/examples/**`, `docs/src/user_guide/**`,
`docs/src/capability_catalogue.md` and `docs/src/TypeHierarchy.md` are written by the docs build
and are not in the tree. Their sources are the Literate files and `docs/capability_catalogue.jl`,
and a defect in one is fixed at its source.

**Markup and code.** A `#src` line, which is an authoring note and not a rendered page, and which
the process-citation census skips for the same reason. A line of Julia. A `##` comment in a
Literate source, which renders inside a code cell. A fenced code block, an `@docs` block, an inline
code span, an inline LaTeX expression, and the target of a markdown link.

---

## The Gate

[`test/test_72_prose_census.jl`](../../test/test_72_prose_census.jl) reads the prose of all four
corpora and holds each text to its row in
[`code_health/prose_baseline.toml`](../../code_health/prose_baseline.toml). A count may fall and
may not rise. A text whose count stands above its row fails, a text with a count above zero and no
row fails, and a text whose every count is zero carries no row, so the baseline empties as the
texts are rewritten. The reader is [`code_health/prose.jl`](../../code_health/prose.jl), which the
census includes rather than copies. The rules the census cannot read hold by review, in the sense
of [`STANDARDS.md`](../../STANDARDS.md).

**Three shapes of text feed one set of counters.** A Literate source gives its `#= … =#` blocks and
every comment line that opens with a `#` and a space at column zero. A Markdown page gives its own lines. The catalogue gives the
body of every double-quoted string literal on a line that is neither a `#` comment nor part of a
triple-quoted block, which is how
[`test/test_71_process_citation_census.jl`](../../test/test_71_process_citation_census.jl) reads
the same file. A triple-quoted block there documents `Cap`, `Section` and `Group` to a contributor
and never renders.

**What it counts.** One column per rule, named in the row: `emdash` and `endash` for rule 13,
`curly` for rule 19, `notjust` for rule 9, `aivocab` for rule 7, `fancy_is` for rule 8, `filler`
for rule 23, `metaphor` for rule 26, `plainword` for rule 31, `bold_label` for rule 16,
`title_case` for rule 17 and `emoji` for rule 18. Rule 7 and rule 23 are read whole. Rule 8 is read
without "features". Rule 26 is read without "vector", "surface", "primitive", "harness" and
"ratchet". Rule 31 is read without "leverage" and "leveraged". `glossary` counts the multi-word
bold terms of [`CONTEXT.md`](../../CONTEXT.md) in their capitalised form, read off that file at
every run so the list never goes stale, and `mechanism` counts the strings of *A page names a type
by its identifier and a concept in plain words*. `verdict` counts the strings of *A check is a
number the reader reads, never a verdict*. `words` records the prose word count of the text and
carries no limit.

**Why those words are exempt.** This library writes "the expected returns vector", "a Pareto
surface", "a feature matrix" and "a leveraged portfolio", and each of those is the concrete word
the rule asks for. The rules still hold on the other sense, and a reader applies them, as rules 10,
11, 27, 28 and 32 and the self-audit are applied.

**Scanning one file.** Run

```bash
julia --project=code_health code_health/prose.jl scan <file>...
```

before and after a rewrite. It prints each file's counts and the row the baseline would carry for
it, and it measures and writes nothing else. Paste that row into the baseline for the files you
rewrote, and delete the row of a file whose counts all reached zero. Do not run `refresh`, which
writes the whole file: rewrite tickets run in parallel, and two sessions that each write the whole
file lose one of the two writes.
