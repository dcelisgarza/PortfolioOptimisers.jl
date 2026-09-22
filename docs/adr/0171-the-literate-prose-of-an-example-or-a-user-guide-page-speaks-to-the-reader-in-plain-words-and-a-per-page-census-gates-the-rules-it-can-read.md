---
status: proposed
---

# The Literate prose of an example or a user-guide page speaks to the reader in plain words, and a per-page census gates the rules it can read

## Context

The examples and the user guide are built by Literate from 73 Julia sources, 62 under `examples/`
and 11 under `user_guide/`. A reader runs the cells of a page in order, and the prose between the
cells says what the next cell does and what the last output shows. Measured against `dev` at
`3b101db2f3`, that prose is 84,479 words.

[ADR 0169](0169-the-docstring-standard-gains-the-unslop-pass-and-a-swept-file-owes-it.md) made the
`unslop` skill the Authority for the patterns of generated text, and it put the Literate sources
outside its Scope, because the pages raise questions the skill does not rule on. Map
[#1218](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1218) was charted for those
pages, and its decision ticket is
[#1219](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1219). The seed is
[#1209](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1209).

Four measurements shaped the decision.

**The lexical tells fall on the old pages as well as the new.** The corpus holds 1,058 lines with
an em dash, about 770 of them on the pages that predate the three pull requests the seed names. It
holds 55 lines with an en dash, none of them spaced, and no curly quote. Rule 13 therefore reaches
the whole corpus, not the new pages alone.

**The new pages and the old pages speak in different voices.** Per 1000 prose words, the old pages
say "we", "you" and "your" 14.0 times and the new pages 1.8. 48 of the 73 pages carry a
`!!! tip "When to reach for this"` admonition, which addresses the reader directly.

**The contributor glossary leaks into the pages.** The prose holds `carrier` 37 times over 10
pages, `read-out` 21 times, `seam` 12, `to the bit` 9, `refused by name` 14, the section sign `§`
58 times over 12 pages, and the capitalised concept names of `CONTEXT.md` 80 times over 16 pages.
The docs site has no glossary page, so a capitalised term points nowhere for a reader.

**Three of the skill's word lists fire on this library's own vocabulary.** Rule 26's list gives 148
hits, dominated by `vector` in "the expected returns vector" and `surface` in "a Pareto surface".
Rule 31 gives 16, of which 13 are `leverage` or `leveraged` in the finance sense. Rule 8 gives 14,
mostly the noun `features` in "a feature matrix". A regular expression cannot read the sense of
those words, so a census over the lists as written can never reach zero on correct prose.

## Decision

**1. Every rule of the `unslop` skill applies as written to all 73 pages.** Rules 20 and 22
describe a chat reply and are waived, as ADR 0169 waived them. The skill stays the Authority: the
instruction file cites a rule by the number the skill gives it and copies no rule.

**2. The prose carries no en dash.** A compound name takes a hyphen, `Black-Litterman`. A numeric
range takes a hyphen or the word `to`. Rule 13 already removes every em dash, and this states the
reading of the same rule for the other dash. The docstrings and the bibliography keep their en
dashes until the sweep of
[#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) adopts the same reading.

**3. The page speaks to the reader.** `you` names what the reader does, chooses, or reads off an
output. `we` names what the page's cells do. The impersonal third person names what the library
does. No census reads this rule.

**4. A page names a type by its identifier and a concept in plain words.** A type is its identifier
in a code span with an `@ref`. A concept is a plain phrase in lower case, and the page defines it
in the sentence where it first appears. A word for the mechanism of the code never appears. The
census holds at zero the multi-word bold terms of `CONTEXT.md` in their capitalised form, and the
exact strings `seam`, `carrier`, `read-out`, `to the bit`, `refused by name` and `§`.

**5. A page shows a check as a number, never as a verdict.** A page can run a comparison and print
its result, and the prose says what the cell computes and what the printed number means for the
reader. The prose never states the outcome as proved. The census holds at zero `to the bit`,
`by construction`, `honest`, `the identity`, `each identity` and `agrees`.

**6. A paragraph carries one job, and no page carries a word band.** A paragraph between two cells
says what the next cell does, or what the last output shows. It restates neither the code nor the
printed result. The census records the prose words of a page as a column with no limit.

**7. The tip, the outline and the take-away are permitted, and none is required.** The
`!!! tip "When to reach for this"` admonition, a numbered outline of the page's sections under the
H1, and a closing `## What to take away` section are page furniture a long page is expected to
carry and a short page does without. Rule 25 binds the take-away to facts the page measured.

**8. The gate is a per-page baseline under `code_health/`.** The build ticket writes a census test
under `test/`, numbered after the last one on `dev`, and `code_health/literate_prose_baseline.toml`
beside the four baselines that file already holds. The baseline is generated, carries a
`[provenance]` block and one row per page, and records every counted rule and the prose word
count. A count may fall and may not rise. A page with no row fails, and a page whose every count is
zero needs no row. Fourteen rewrite tickets run in parallel, and a per-page row keeps them off one
shared number.

**9. The census counts the rules it can read, less the words this library owns.** It counts rules
13, 19, 9, 16, 17 and 18, rules 7 and 23 whole, rule 8 without `features`, rule 26 without
`vector`, `surface`, `primitive`, `harness` and `ratchet`, and rule 31 without `leverage` and
`leveraged`, plus the strings of decisions 4 and 5 above. The exempt words stay under the skill and
a reader applies them, as rules 10, 11, 27, 28, 32 and the self-audit are applied. The census skips
a `#src` line, as the process-citation census does, and it reads the `Description` line of the
` ```@meta ` block. Its glob is `examples/**/*.jl` and `user_guide/*.jl`.

**10. The rule lives in its own instruction file.**
`.github/instructions/julia-literate-prose.instructions.md` carries the glob
`examples/**/*.jl, user_guide/*.jl`, so the rule attaches to the file a rewrite session edits.
`STANDARDS.md` gains a row per rule subject, and the Scope sentence of
`julia-docstrings.instructions.md` points at the new file.

## Considered options

**The lexical rules on the 15 new pages alone.** Rejected: the older pages carry the tells at a
higher rate, so the map's destination sentence would be false and the census would need two scopes.

**The en dash kept in a compound name.** Rejected by the maintainer: the prose carries no dash of
either kind, and one reading is simpler than a rule that admits an en dash between two surnames
and refuses it elsewhere.

**The impersonal third person.** Rejected: 48 pages open with a tip that tells the reader when to
reach for the estimator, and that sentence has no actor without the second person. Rule 29 would
then push those sentences back toward the passive.

**The capitalised glossary term, defined on first use.** Rejected: the docs site has no glossary
page, so the capital points nowhere for a reader, and "defined on first use" is a rule no census
reads, which leaves 80 hits outside the gate.

**A word band of 1,500 prose words per page, or 250 per section.** Rejected: a page is long because
it covers more ground or because its paragraphs repeat the cells, and a band punishes both the
same. The paragraph rule names the defect, and the word column records the corpus shrinking.

**A library-wide total per rule.** Rejected: two parallel rewrite tickets each lower one number,
and the merge keeps one of them, which `CLAUDE.md` names as the failure of a shared ratchet.

**A section of `julia-docstrings.instructions.md`.** Rejected: that file's glob is
`src/**/*.jl, ext/**/*.jl, docs/**/*.md`, which matches no Literate source. Widening it would
attach 887 lines of docstring templates to an example page, and leaving it attaches nothing to the
file a rewrite session edits.

**Counting every word of every list, with a per-page exemption.** Rejected: `vector` alone appears
on many pages in its literal sense, so the exemption file would be long, and every row of it would
record correct prose.

## Consequences

**Zero stays reachable, and the map can close on it.** Every counted rule can reach zero on every
page without a sentence becoming wrong, so the map's closing condition holds as written.

**The pages and the docstrings disagree on two points for a while.** A page writes
`coverage universe` in lower case where a docstring writes `Coverage Universe`, and a page writes
`Black-Litterman` where the bibliography writes an en dash. The sweep of #404 owns both, and this
ADR records the reading it would adopt.

**A rewrite ticket scans its own pages.** The census exposes a function that reads one file, so a
session measures a page before and after its rewrite without running the suite.

**A new page owes a row.** A page added after the census lands fails until its row is written or
its counts are zero, which is the same shape as the four baselines under `code_health/`.

**The catalogue and the Markdown pages stay outside.** The `Prose` text of the Capability
Catalogue, `docs/src/migration.md`, `README.md` and `docs/src/index.md` carry the same voice and
are held as fog on #1218. The census glob can widen to them when that fog is ticketed.
