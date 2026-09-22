---
status: proposed
---

# The docstring standard gains the `/unslop` pass, and a swept file owes it

## Context

The sweep of [#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) holds every
file in `src/` and `ext/` to three conditions: its documentation states the mathematics, its code
agrees with that statement when it is run, and its lines are covered or exempted. On 2026-09-22,
115 of the 321 rows of `code_health/sweep_manifest.toml` read `swept = true`.

The prose of those files was written by the same sessions that wrote the mathematics, and it
carries the patterns of generated text. The `unslop` skill names those patterns as numbered rules.
Measured against `dev` at `3b101db2f3`, the swept files hold 713 em dashes over 70 of the 115
files, which is rule 13 of the skill alone, and `src/` and `ext/` together hold 2078 over 204
files. The lexical rules are the small part of the skill. Rules 27, 28 and 32, which ask a sentence
to name a mechanism, to carry one idea, and to drop a flourish, are the part that a reader pays
for, and no parser reads them.

A sibling map, [#1218](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1218), was
charted the same day for the 73 Literate pages of the examples and the user guide. It puts every
docstring out of its scope and names #404 as the owner. Its decision ticket,
[#1219](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1219), decides the rule for
the Literate prose, and it holds the `Prose` text of the Capability Catalogue as fog.

The maintainer asked for `/unslop` in the docs and docstrings standard and in the docs sweep, and
said that this reopens the sweep's documentation tickets.
[#1236](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1236) is the charter.

## Decision

**1. The `unslop` skill is the Authority for the patterns.**
`.github/instructions/julia-docstrings.instructions.md` gains § *The prose passes `/unslop`*. The
section cites the skill by name, copies none of its rules, and names a rule by the number the skill
gives it. Two live copies drift, and the skill's numbering is stable by its own statement.

**2. The Scope is the standard's own, less the Literate sources.** The rule reaches the docstrings
of `src/**/*.jl` and `ext/**/*.jl`, the dictionary values of
`src/01_Base/01_DocstringDictionaries.jl` that they interpolate, and the hand-written Markdown
pages under `docs/src/` outside `docs/src/contribute/`. The Literate sources and the catalogue's
`Prose` stay outside it, because #1219 decides them, and a rule written here first would either
duplicate that decision or contradict it.

**3. Three points of contact with the standard are stated, and no more.** A template of the
standard fixes the shape of a section, and the skill's rules apply to the sentences inside it, so
a `# Fields` bullet stays a list entry. A dictionary value is rewritten once, in its dictionary.
Rules 20 and 22 describe a chat reply and never arise. Every other rule applies as written.

**4. The pass is the fourth condition of the sweep.** #404's Destination and rules state four
conditions, each of the thirteen child maps states the fourth, and `code_health/sweep_triage.jl`
writes four conditions into a new sub-issue. A file added to a swept map owes the pass as it owes
the other three.

**5. The swept rows keep `swept = true`, and the tickets carry the pass.** The flag arms the gates
of the docstring standard: the `algorithm` floor, the zero `# Details`, the `# Related` on a
dispatch alias, the `math_dict` interpolation. A flip to `false` would drop those gates and reopen
work that is done. ADR 0085 chose the same when `# Details` was abolished after eight files were
swept. The manifest header now states what the flag records, and the manifest gains no key.

**6. The closed documentation tickets are reopened, not replaced.** Every closed `Document …` and
`Sweep …` sub-issue of a child map whose files are swept, 56 tickets, and the one closed child map, #418, are reopened
under #1236. Each carries the pass for the files it swept, under the paths the tree gives them
today. A `Check and cover …` ticket is coverage work and stays closed; a grilling or a bug ticket
is not a sweep ticket and stays closed. Six closed documentation tickets under child maps 9 and 11
documented files whose rows still read `swept = false`, and they stay closed too: the open sweep
ticket of each such file carries the pass with the other three conditions. The tickets were
reopened rather than replaced because
each already names its files, its reference set, and the traps its sweeper found, and a new ticket
would copy all of that or lose it.

**7. There is no Gate.** The row in `STANDARDS.md` reads `none — unenforced`. The rule holds by
review, in the sense of that file, and the sweep ticket is the record that the review happened.

## Considered options

**A gate on the lexical rules now.** A census in the `test_26` idiom could count em dashes and the
word lists of rules 7, 8, 23, 26 and 31 over `src/` and `ext/`, hold a swept file at zero, and hold
the library total as a ratchet. Rejected for now: a swept file at zero would red the build on 70
files on the day it lands, and a library ratchet is one number that parallel sessions each lower,
which `CLAUDE.md` warns against. #1220 builds a census for the Literate prose with per-page rows,
and that shape can widen to docstrings once it exists. The measurement above is the baseline for
that decision.

**Flip the 115 rows to `swept = false`.** Rejected, as in ADR 0085: the flag is the ratchet for
the documentation standard, and the coverage and defect passes still hold.

**A `unslop` key per manifest row.** Rejected: no Julia test can read the rules, so the key would
record a claim that nothing checks, and the ticket already records it.

**One new sub-issue per swept file.** Rejected: 115 new tickets would restate what 56 closed
tickets already hold, and a per-file match of the swept files to those tickets is not possible by
name, because #1058 split files as well as moving them.

**A rule that also covers the Literate sources.** Rejected: #1219 decides voice, vocabulary and
length for those pages, which the skill does not rule on, and its answer may live in the same
instruction file. The section here leaves that room.

## Consequences

**The reference docstrings are not yet compliant.** The table in the standard names units in
`src/04_MatrixProcessing/02_Denoise.jl` and `src/06_Distance/02_Distance.jl`. Both files are swept
and both are re-read under the reopened tickets of child map 2. A reader of the table sees a
pointer before the pass, as ADR 0085 accepted for its own window.

**A sweep ticket has one more step, and the last step.** The pass runs after the mathematics is
checked, because a rewrite of a sentence that is about to change is wasted. A ticket's resolution
names what the skill's self-audit found and changed.

**The docstring sweep and the Literate sweep meet at the standard.** If #1219 puts the Literate
rule in `julia-docstrings.instructions.md`, the two sections sit side by side and the Scope
sentence of this one already names the boundary.

**A dictionary value changes many docstrings at once.** A pass over
`src/01_Base/01_DocstringDictionaries.jl` rewrites text that every interpolating docstring
renders, so that file's ticket runs before the others of its map, or its rewrites are checked
against a rendered page.
