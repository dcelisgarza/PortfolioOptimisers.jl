---
status: proposed
---

# The prose of every user-facing page speaks to the reader in plain words, and a per-text census gates the rules it can read

## Context

A user of this library reads four bodies of prose, and no two of them live in the same kind of
file.

- **73 Literate sources** under `examples/` and `user_guide/`, 62 and 11, from which the docs build
  renders the examples and the user guide. A reader runs the cells of such a page in order, and the
  prose between the cells says what the next cell does and what the last output shows. Measured
  against `dev` at `3b101db2f3`, that prose is 84,479 words.
- **642 hand-written Markdown pages** under `docs/src/` outside `docs/src/contribute/`: the landing
  page, the migration guide, the API index, the bibliography page, and 638 mirror pages under
  `public_api/` and `private_api/` that host the docstrings.
- **`README.md`**, the repository front page, and the one user-facing Markdown file at the root.
  `CLAUDE.md`, `AGENTS.md`, `CONTEXT.md`, `STANDARDS.md`, `CODE_OF_CONDUCT.md`, `.github/` and
  `research/` are written for a contributor, as `docs/src/contribute/` is.
- **`docs/capability_catalogue.jl`**, a Julia data file from which the docs build renders the
  Capability Catalogue. It holds 114 `Prose` paragraphs, 45 `Note` texts, 94 `Cap` labels and 76
  `Section` titles.

[ADR 0169](0169-the-docstring-standard-gains-the-unslop-pass-and-a-swept-file-owes-it.md) made the
`unslop` skill the Authority for the patterns of generated text. Map
[#1218](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1218) was charted for the
Literate pages, its first decision ticket is
[#1219](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1219), and the seed is
[#1209](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1209). #1219 ruled the
Literate pages and left the other three corpora as fog. Ticket
[#1239](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1239) graduated that fog and
widened this ADR, which had not reached `main` and was therefore still a draft.

### What the Literate pages measured

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

### What the other three corpora measured

Measured with the census reader itself at `76d82e3e64`, not with `grep`, so these are the numbers a
widened census prints.

| text | prose words | emdash | endash | filler | bold_label | glossary | mechanism | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `docs/capability_catalogue.jl` | 7,028 | 4 | 1 | 1 | 0 | 38 | 8 | 2 |
| `docs/src/migration.md` | 2,598 | 7 | 1 | 0 | 5 | 9 | 2 | 1 |
| `docs/src/index.md` | 805 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| `README.md` | 476 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| `docs/src/00_API.md` | 223 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `docs/src/99_references.md` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 638 mirror pages | 19,687 | 51 | 4 | 8 | 2 | 222 | 50 | 2 |

**The catalogue is the outlier, and its 38 glossary hits are the work.** They are `Asset Panel` and
`Panel Field` five times each, then `Factor Family Basis`, `Hindsight Comparator`, `Online Scheme`
and twenty more. Its 8 mechanism words are `carrier` four times, `read-out` three times and one
`§`. `README.md`'s 6 em dashes all sit in the feature bullets, not in the two-sentence opening that
`test/test_64_docs_page_metadata_census.jl` holds identical to the landing page.

**A mirror page's hits are mostly its own name, written twice.** Of its 222 glossary hits, 74 are in
the H1, such as `# The Asset Panel: private API`, whose shape ADR 0128 fixes, and a further 72 are
in the `Description` line, which `docs/page_metadata.jl` derives and `test_64` fails when it
drifts. With both exempt, the 638 pages fall from 417 counted hits over 186 pages to 181 over 67,
across 9,814 words of body prose.

### What the code of a page measured

The rewrite tickets changed prose alone, and they found the same tells in the code cells, where no
rule reached. Ticket [#1299](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1299)
measured them with the widened census reader at `98db9442a0`.

- **253 comments in the code of 47 texts.** 44 are Literate sources, and the other three are
  `README.md`, `docs/src/index.md` and `docs/src/migration.md`. Most say what the next line does
  ("Compute the returns", "Format for pretty tables"). A few are empty `#` marks that hold a line
  break for the formatter. A few are gotchas, such as a radius that sits on a knife edge.
- **30 counted hits in printed strings over 15 texts.** 12 em dashes in `pretty_table` titles,
  capitalised glossary terms ("The Span Rule", "filled with the Held Price", "The synthetic Asset
  Panel"), the verdict word "agree", and title-case titles.
- **About 470 lines that print a string.** The census counts the tells, but the larger defect is a
  title that states a claim: "Both knobs move the clustering", "Tighter tracking-error budget hugs
  the benchmark". A title is read first, and a reader takes it as the finding of the cell.

## Decision

**1. The rule governs every text a user reads as a page.** Four corpora: the 73 Literate sources
under `examples/**/*.jl` and `user_guide/*.jl`, every hand-written Markdown page under
`docs/src/**/*.md` outside `docs/src/contribute/`, `README.md`, and `docs/capability_catalogue.jl`.
A docstring is outside it: it reaches a page through `@docs`, and ADR 0169 owns it. `README.md` is
the only user-facing Markdown file at the repository root.

**2. Every rule of the `unslop` skill applies as written.** Rules 20 and 22 describe a chat reply
and are waived, as ADR 0169 waived them. The skill stays the Authority: the instruction file cites
a rule by the number the skill gives it and copies no rule.

**3. The prose carries no en dash.** A compound name takes a hyphen, `Black-Litterman`. A numeric
range takes a hyphen or the word `to`. Rule 13 already removes every em dash, and this states the
reading of the same rule for the other dash. The docstrings and the bibliography keep their en
dashes until the sweep of
[#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) adopts the same reading.

**4. The page speaks to the reader.** `you` names what the reader does, chooses, or reads off an
output. `we` names what the page's cells do. The impersonal third person names what the library
does. No census reads this rule.

**5. Three rules read a page a reader runs, and a text with no cells is outside them.** They are
`we` for the page's cells, decision 8 below, and decision 9 below. Every other rule of this ADR
holds on all four corpora, `you` for the reader and the impersonal third person for the library
included. A mirror page, `README.md` and the catalogue carry no cells and owe the other rules in
full.

**6. A page names a type by its identifier and a concept in plain words.** A type is its identifier
in a code span with an `@ref`. A concept is a plain phrase in lower case, and the page defines it
in the sentence where it first appears. A word for the mechanism of the code never appears. The
census holds at zero the multi-word bold terms of `CONTEXT.md` in their capitalised form, and the
exact strings `seam`, `carrier`, `read-out`, `to the bit`, `refused by name` and `§`.

**7. A page shows a check as a number, never as a verdict.** A page can run a comparison and print
its result, and the prose says what the cell computes and what the printed number means for the
reader. The prose never states the outcome as proved. The census holds at zero `to the bit`,
`by construction`, `honest`, `the identity`, `each identity` and `agrees`.

**8. A paragraph carries one job, and no page carries a word band.** A paragraph between two cells
says what the next cell does, or what the last output shows. It restates neither the code nor the
printed result. The census records the prose words of a text as a column with no limit.

**9. The tip, the outline and the take-away are permitted, and none is required.** The
`!!! tip "When to reach for this"` admonition, a numbered outline of the page's sections under the
H1, and a closing `## What to take away` section are page furniture a long page is expected to
carry and a short page does without. Rule 25 binds the take-away to facts the page measured.

**10. Derived text is outside the rule.** The rule reads written prose. A text that a script
derives, or that another census holds to a shape, is outside it, because two gates over one line
disagree sooner or later. Two such texts exist: the H1 of a mirror page under
`docs/src/public_api/` or `docs/src/private_api/`, whose shape ADR 0128 fixes, and that page's
`Description` line, which `docs/page_metadata.jl` derives and `test_64` gates. The path tells a
mirror page from a written one, as `test_64` already does. The `Description` line of a Literate
page and of a hand-written page is written prose, and the rule reads it. The four generated pages
are not in the tree at all, and the census skips them, as the process-citation census does.

**11. The catalogue's prose is every double-quoted string literal on a non-comment line.** The file
holds no `Cap` name that is a string, so every such literal is text the catalogue renders. The
census keeps the design it states in its own header: it reads text, loads only `TOML`, parses no
Julia and loads no package under measurement.
`test/test_71_process_citation_census.jl` already reads this file the same way, with
`page_hits(; comments = false)`.

**12. The gate is one per-text baseline under `code_health/`.** One table holds every text of all
four corpora, keyed by path. The baseline is generated, carries a `[provenance]` block and one row
per text, and records every counted rule and the prose word count. A count may fall and may not
rise. A text with a count above zero and no row fails, and a text whose every count is zero carries
no row. Rewrite tickets run in parallel, and a per-text row keeps them off one shared number.

**13. The census counts the rules it can read, less the words this library owns.** It counts rules
13, 19, 9, 16, 17 and 18, rules 7 and 23 whole, rule 8 without `features`, rule 26 without
`vector`, `surface`, `primitive`, `harness` and `ratchet`, and rule 31 without `leverage` and
`leveraged`, plus the strings of decisions 6 and 7 above. The exempt words stay under the skill and
a reader applies them, as rules 10, 11, 27, 28, 32 and the self-audit are applied. The census skips
a `#src` line, a fenced block that is not code, an `@docs` block, an inline code span, an inline
LaTeX expression and the target of a markdown link. Decisions 16 and 17 state what it reads in the
code.

**14. The rule lives in one instruction file, and no artifact of the rule is named for Literate.**
`.github/instructions/julia-prose.instructions.md` carries the glob
`examples/**/*.jl, user_guide/*.jl, docs/src/**/*.md, README.md, docs/capability_catalogue.jl`, so
the rule attaches to the file a rewrite session edits. Its Scope sentence excludes
`docs/src/contribute/`, as the docstring standard's does. The build ticket renames the three
remaining artifacts to `code_health/prose.jl`, `code_health/prose_baseline.toml` and
`test/test_72_prose_census.jl` in the commit that widens them, so a rewrite session in flight loses
its `scan` path once and not twice. ADR 0169 narrows to the docstrings and the dictionary values
they interpolate, and its Scope sentence names this ADR as the owner of every page.

**15. The work joins map #1218.** The map's destination widens from the 73 Literate pages to every
text this ADR governs. One build ticket widens the rule's three code artifacts and seeds their
rows; four rewrite tickets follow, for the catalogue, for the four top-level pages with `README.md`,
for the `public_api` mirror pages and for the `private_api` mirror pages. The verify ticket gains
the new texts.

**16. A string that a cell prints is prose.** Ruled by the maintainer on 2026-09-24. A
`pretty_table` `title` or `source_notes`, a plot `title`, `label`, `xlabel`, `ylabel` or
`colorbar_title`, a `println` sentence, and a column name or a cell that a table prints all render
as text that a reader reads, and every rule of this ADR holds on them. A title says what its table
or plot shows, and it makes no claim that the output of its cell does not show. A string that code
reads as data stays as the code needs it. The census reads the body of every plain double-quoted
string literal in the code of a page: every code line of a Literate source, and the body of every
`julia`, `@example`, `@repl` and `@setup` fence of a Markdown page or of the prose of a Literate
source. It reads an interpolation as a
space, and it reads a string that a `title =` keyword opens as a heading too, so rule 17 reaches a
title. The reader is a lexer of one line at a time, not a parser, so decision 11's design holds.

**17. A comment in a code cell is removed, unless it is a gotcha, and a gotcha is a `#!` line.**
Ruled by the maintainer on 2026-09-24. A gotcha is a thing that a reader who copies the cell gets
wrong without the comment. It is short, one line is the target, and it sits on its own line above
the code it is about. Its mark is `#!` and a space. Literate's `ismdline` makes a markdown line only
of a bare `#`, or of a `#` and a space followed by text, so a `#!` line stays in the code cell and renders as
written, where a `##` line renders with one `#` removed. Literate reads `#!md`, `#!nb` and `#!jl`
at the start of a line as negated filter tokens and deletes the line from one output, so the space
is part of the mark. The census reads the text of a gotcha as prose, and counts every other
comment in a new column, `comment`, which holds at zero. Three markers are markup and not
comments: Literate's `#-` and `#+`, and Documenter's `# hide`.

**18. A cell that prints a verdict, and a section that checks the library, stay.** Ruled by the
maintainer on 2026-09-24. A cell that prints `true`, `yes` or a `Binds?` column, and the sections of
the online walk-forward page that print how far the online run is from the batch run, illustrate
the point for the reader. Neither changes under decision 7. The words of their label strings follow
decision 16.

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

**Counting every word of every list, with a per-page exemption.** Rejected: `vector` alone appears
on many pages in its literal sense, so the exemption file would be long, and every row of it would
record correct prose.

**The docstring standard keeps the `docs/src/` pages, and this rule takes the catalogue and
`README.md` alone.** Rejected: one file would then carry two prose rules that differ in five
points, which `improve-standards` names as the drift hazard. The division by what a text is, a
docstring against a page, is stable where a division by directory is not.

**A third instruction file holding the rules the two standards share.** Rejected: three files where
two suffice, and the shared list is short enough that the cheaper fix is to move the pages whole.

**The rule stops at the top level of `docs/src/`.** Rejected by the maintainer: a boundary that
excluded the 638 mirror pages would leave 181 counted hits over 9,814 words of hand-written prose
ungated, and a mirror page is a page a user reads.

**The rule reads a mirror page's H1 and its derived `Description`.** Rejected: that text is derived
by `docs/page_metadata.jl` and gated by `test_64`, so the rule would put two gates on one line and
would force ADR 0128's H1 shape to change with it. The cost of reading them is 417 hits over 186
pages against 181 over 67.

**A second baseline for the three new corpora.** Rejected: the baseline is keyed by path already,
so a Markdown page and the catalogue are more keys in the same table. Two stores need two readers
kept in step.

**A new map for the three new corpora.** Rejected: the rule, the gate and the rewrites are one
effort, and #1218's destination sentence is the one that must come true.

**A code comment keeps its `##` mark, and the rule reads its text.** Rejected by the maintainer: a
comment that says what the next line does repeats the code, and the prose above the cell is where
a page explains itself. Only a gotcha earns a place in the cell.

**The gotcha as an admonition in the prose.** Rejected: a reader who copies the cell copies the
code and leaves the prose, and a gotcha must travel with the line it is about.

**The census leaves the printed strings to a reader.** Rejected: the counted rules read a string as
exactly as a paragraph, and a lexer of one line costs about a hundred lines of text handling. Only
the claim a string makes is left to a reader, as it is for a paragraph.

**The census reads only the strings of known keywords, `title`, `label` and `println`.** Rejected:
a column name and a cell of a printed table render as well, and a keyword list goes stale when a
page calls a new plotting function. A string that code reads as data carries no tell, so reading
every plain string costs no false hit.

**`include`ing the catalogue and walking its nodes.** Rejected: the census would compile 1,520
lines and define `Cap`, `Prose` and `CATALOGUE` beside `test/test_26_docs.jl`, which includes the
same file. A text reader costs nothing and matches the census's stated design.

## Consequences

**Zero stays reachable, and the map can close on it.** Every counted rule can reach zero on every
text without a sentence becoming wrong, so the map's closing condition holds as written.

**The gate can widen before the rewrites land.** A text enters the baseline at the counts it
carries today, and a count may fall and may not rise, so the census does not red on the day the
build ticket widens it.

**The pages and the docstrings disagree on two points for a while.** A page writes
`coverage universe` in lower case where a docstring writes `Coverage Universe`, and a page writes
`Black-Litterman` where the bibliography writes an en dash. The sweep of #404 owns both, and this
ADR records the reading it would adopt.

**A mirror page's prose and the docstrings it hosts are rewritten by different tickets.** The lead
sentence of a mirror page is this map's, and every docstring the page hosts is #404's. The two must
not contradict each other, so a mirror rewrite reads the page as rendered.

**A rewrite ticket scans its own files.** The census exposes a function that reads one file, so a
session measures a text before and after its rewrite without running the suite.

**A new page owes a row.** A page added after the census widens fails until its row is written or
its counts are zero, which is the same shape as the four baselines under `code_health/`.

**A rewrite can now change a line of Julia, but only its strings and its comments.** The check
that a rewrite changed no code strips every comment and empties every string literal, then compares
the tokens of each code cell before and after. It compares tokens and not lines, because the
formatter can reflow a call once an empty `#` that held its line break is gone.

**A string a test reads changes with the test.** A title or a label that a test or a doctest reads
is changed in the same commit as the test.

**A code cell can hold a multi-line string only at a cost.** The lexer reads one line at a time.
No code cell holds a triple-quoted or a multi-line string today. A page that adds one would need a
lexer that carries its state from line to line.

**Three artifacts change name once.** The build ticket renames `code_health/literate_prose.jl`,
`code_health/literate_prose_baseline.toml` and `test/test_72_literate_prose_census.jl` in the same
commit that widens them, and it edits the open rewrite tickets that name the old `scan` path.
