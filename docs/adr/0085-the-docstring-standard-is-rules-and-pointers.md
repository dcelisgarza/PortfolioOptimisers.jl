---
status: accepted
---

# The docstring standard is rules and pointers, and `# Details` is abolished

## Context

[#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) sweeps every file under
`src/` and `ext/` for three things at once. Thirteen child maps carry the work. Eight files are
swept today, and about 28 prose tickets are open across child maps 1 to 4.

[#478](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/478) opened on the observation
that the swept docstrings do not share one language. `src/05_Denoise.jl` states the noise condition
three ways across three siblings of one family. `src/05_Denoise.jl` and `src/06_Detone.jl` each
carry a numerical experiment on an unnamed sample. None of that breaks a rule, because no rule
covers it.

Five defects were measured before this decision was taken.

 1. **`# Details` has no Authority rule.** The Authority mentions it four times, and every mention
    is inside a template. The only text that describes it is the placeholder
    `Additional implementation notes.` 299 docstrings over 84 files carry the section.
 2. **The `## Complete Example` is fictional and self-contradicting.** It runs 380 lines, which is
    37 % of the Authority. It documents a `MyAbstractCustomProcess` family that does not exist, so
    no gate can ever read it. It appends ``in `PortfolioOptimisers.jl` `` to a summary sentence at
    two places, which the same file's load-bearing summary rule forbids. It interpolates no
    dictionary key at all, although `CLAUDE.md` states that field text is centralised.
 3. **A second live copy of the templates sits in the prompts.**
    `.github/prompts/julia-docstrings.prompt.md` runs 292 lines and duplicates four whole
    templates. Nothing links it by name. The three `add-*` prompts duplicate more. The four files
    carry 40 docstring section headings between them. `STANDARDS.md` forbids exactly this.
 4. **Four cited names are dead.** `AbstractDetoneAlgorithm` and `AbstractPosdefAlgorithm` in
    `add-algorithm.prompt.md`, `AbstractLowOrderPriorResult` in `add-result.prompt.md`, and
    `ClustersResult` in `julia-source-code.instructions.md`. None exists in `src/` or `ext/`.
 5. **`STANDARDS.md` carries a stale measurement.** Its Precedence section states that `dev` holds
    25 ADRs that `main` has never seen, measured at `9adac7735b`. `origin/main` now stands at
    `cde5bf48cc`, and every ADR is on it. The paragraph counts zero.

`math_dict` holds 20 keys and is interpolated 343 times, so most symbols are written by hand at
every site. `\lambda_+` is written inline 16 times in `src/05_Denoise.jl` alone, and it has no key.
That is the mechanism behind the family drift that opened the issue.

## Decision

The Authority for every rule below is
`.github/instructions/julia-docstrings.instructions.md`, except where a rule names another file.
`STANDARDS.md` routes to it.

**1. The Authority carries rules and pointers, not worked examples.** The 380-line
`## Complete Example` is deleted. A `## Reference docstrings` table replaces it, with one row per
docstring kind, and each row names a real unit in a file whose sweep-manifest row reads
`swept = true`. A pointer cannot drift, because a gate holds its target. The small per-section
templates stay, because a section's shape and order is the rule in normative form. The file gains a
`## Vocabulary` section, in the shape of the one that `STANDARDS.md` already carries.

**2. `# Details` is abolished.** The section holds facts that five other sections already own, and
it holds them because nothing said where they belonged. Each fact moves by its subject:

- A fact about one field moves to that field's description.
- A fact about a raise or a precondition moves to `# Validation`.
- A fact about the type as a whole moves to the summary paragraph, in its second sentence or later.
  The Capability Catalogue extracts the first sentence only, so a later sentence never reaches it.
- A fact about another unit moves to `# Related`, as an annotated entry.
- A mis-filed step moves to `# Algorithm`, an argument contract to `# Arguments`, the shape of a
  result to `# Returns`, and a model row to `# JuMP formulation`.

**3. The abolition is gated by two checks in `test/test_26_docs.jl`.** A file whose sweep-manifest
row reads `swept = true` carries zero `# Details` sections. The library-wide total may not rise
above a single recorded number, which stands at 299 today and falls with each sweep. The second
check retires when the number reaches zero. The first check mirrors the `# Algorithm` floor of
[ADR 0081](0081-the-docstring-standard-states-the-model-it-builds.md) and inverts it. That floor
may not fall. This total may not rise.

**4. The `# Mathematical definition` boundary points both ways.** The section names no identifier
from the body, states no order of operations, and states no property that the implementation chose
rather than the mathematics. A mathematical consequence of the definition stays. ADR 0081's
amendment of the same date records the reverse direction, which that decision already held.

**5. Notation is fixed by symbol and by family.** A symbol that appears in the docstrings of two or
more units becomes a `math_dict` key in `src/01_Base.jl`, and every site interpolates it. Siblings
of one leaf abstract supertype state the same quantity in the same form. The library holds 160
families under that definition, and 137 of them fit inside one file, so the check is local to a
sweep ticket.

**6. A prompt holds no docstring template.** `.github/prompts/julia-docstrings.prompt.md` is
deleted, because nothing links it by name and all three `add-*` prompts already carry its unique
steps. Each template in `add-estimator.prompt.md`, `add-algorithm.prompt.md` and
`add-result.prompt.md` becomes a link to the Authority section that owns it.

**7. A sweep ticket routes to the standard.** `body_of` in `code_health/sweep_triage.jl` writes a
fixed `## Routing` block that names `STANDARDS.md`, the Authority and `CONTEXT.md` directly. The
same block is added by hand to the open tickets. Of six tickets read, three named the Authority and
none named `STANDARDS.md`, and the only existing route was one hop through a 5111-word umbrella
issue.

**8. A new census gates the two staleness classes.**
`test/test_46_standards_citation_census.jl` checks that every backticked identifier and every path
cited in a standards file resolves, against a named allow-list of deliberate placeholders. It also
checks that no standards file states a bare count of the repository. It joins `test_40` through
`test_45` in that idiom. It fails on the four dead names and on the stale Precedence paragraph
until both are corrected.

**9. The migration is per file, inside each file's own sweep ticket.** A `# Details` section and an
inline symbol move when that file's sweep ticket opens its docstrings. Only the seven swept files
that carry `# Details` migrate up front, and their 29 sections migrate under #478. The other 77
files migrate on their own tickets. No file is touched before its mathematics is checked.

**10. The vocabulary splits by kind.** `Selector tag` names a type role, so it enters `CONTEXT.md`
§ 1, *Core Abstractions*, beside the other abstractions. The process terms — Unit, Family,
Reference docstring, Capability Catalogue, Coverage Exemption — enter the Authority's own
`## Vocabulary` section. `CONTEXT.md` stays a domain glossary and nothing else.

## Consequences

**The prose sweep pauses, and the coverage sweep does not.** Seven children of #478 gate the
resume. About 28 prose tickets across child maps 1 to 4 wait. The check-and-cover tickets touch no
prose, so they keep running. The cost of the pause is bounded by the eight files swept so far. That
number only grows, so the pause is cheaper now than at any later date.

**The eight swept rows keep `swept = true`.** That flag states that a file passed three conditions.
This decision changes the first one only. The defect pass and the coverage pass still hold, so a
flip to `false` would re-open work that is genuinely done and would drop the `algorithm` floor that
only a `swept = true` row carries. The prose of those eight files is re-read under a child of #478
instead. The sweep manifest gains no key and no version stamp.

**The reference table names units that are not yet compliant.** The table is written before the
eight files are re-read. That window sits entirely inside the pause, so no sweep ticket reads a
stale pointer. The closing condition of the re-sweep child is that every unit the table names
complies.

**`# JuMP formulation` gets no reference row yet.** Every file that calls a `JuMP` macro sits in
child map 6, 9, 10 or 11, and none of those files is swept. That row is absent from the table until
one of those maps sweeps a file, and #443 settles the trigger first.

**The split of the 299 sections is not measured.** A crude classifier recognised 191 mis-filed
steps and 105 branch statements out of 955 bullets. The remaining 652 is an upper bound on the
residue, not a measurement of it. The true split is found file by file, which is the reason that
decision 9 places the migration inside the sweep ticket rather than in one library-wide pass.

**Two rules stay unenforced.** Decision 4 and the family half of decision 5 cannot be gated by a
parser, because neither an implementation fact nor an equation's form is a token. They hold by
review, in the sense of `STANDARDS.md`. That is a known state and not a hidden one.
