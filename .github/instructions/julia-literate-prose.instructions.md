---
applyTo: 'examples/**/*.jl, user_guide/*.jl'
---

# Literate Prose Guidelines for PortfolioOptimisers.jl

The examples and the user guide are built by Literate from the Julia sources under `examples/` and
`user_guide/`. This file governs their **prose**: the text of every `#= … =#` block, every comment
line that Literate renders as Markdown, and the `Description` line of a ` ```@meta ` block. A line of Julia, a fenced code
block, a LaTeX expression, an `@ref` link and a `#src` line are outside it.

A reader runs the cells of a page in order. The prose between two cells says what the next cell
does, or what the last output shows.

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
line, an `@ref` link, an admonition's indentation and a fenced block stay as they are, and the
outputs of the page do not move.

---

## The page speaks to the reader

- **`you` names what the reader does, chooses, or reads off an output.** "Reach for `MeanRisk`
  when you want the trade-off between return and risk." "The table you get back holds one row per
  fold."
- **`we` names what the page's cells do.** "We fit the prior on the first training window."
- **The impersonal third person names what the library does.** "`prices_to_returns` carries every
  gap into the returns."

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

A paragraph between two cells says what the next cell does, or what the last output shows. It
restates neither the code nor the number the cell prints.

A page carries no word band. A page is long because it covers more ground, and the paragraph rule
is what keeps it from being long because it repeats itself.

---

## The page's furniture

Three elements are permitted on any page, expected on a long one with many sections, and required
on none.

- The `!!! tip "When to reach for this"` admonition under the H1, which says when a reader reaches
  for the estimator and what to reach for instead.
- A numbered outline of the page's sections under the H1.
- A closing `## What to take away` section. Rule 25 binds it: it states the facts the page
  measured, never a generic close.

A short page carries none of them and is complete without.

---

## The Gate

A census under `test/` reads the prose of every file this file governs and holds each page to its
row in a generated baseline under `code_health/`. A count may fall and may not rise. A page with no
row fails, and a page whose every count is zero needs no row. Until that census lands the rule
holds by review, in the sense of [`STANDARDS.md`](../../STANDARDS.md).

**What it counts.** Rule 13, rule 19, rule 9, rule 16, rule 17 and rule 18. Rule 7 and rule 23
whole. Rule 8 without "features". Rule 26 without "vector", "surface", "primitive", "harness" and
"ratchet". Rule 31 without "leverage" and "leveraged". The strings of the two sections above, and the multi-word bold
terms of [`CONTEXT.md`](../../CONTEXT.md) in their capitalised form. It records the prose word
count of a page as a column with no limit.

**Why those words are exempt.** This library writes "the expected returns vector", "a Pareto
surface", "a feature matrix" and "a leveraged portfolio", and each of those is the concrete word
the rule asks for. The rules still hold on the other sense, and a reader applies them, as rules 10,
11, 27, 28 and 32 and the self-audit are applied.

**What it does not read.** A `#src` line, which is an authoring note and not a rendered page, and
which the process-citation census skips for the same reason. A line of Julia, a fenced code block
and a LaTeX expression.

**Scanning one page.** The census exposes a function that reads one file and prints its counts, so
a rewrite session measures a page before and after its rewrite without running the suite.
