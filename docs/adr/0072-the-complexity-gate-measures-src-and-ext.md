---
status: accepted
---

# The complexity gate measures `src/` and `ext/`, and a Declaration Macro is measured at its declaration

## Context

`CodeComplexity` is a pure `JuliaSyntax` parser. It loads no package and runs no inference, which
is what makes it cheap — 4.8 s over the whole tree. The price is that it reads source, so a macro
that turns a declaration into definitions is a blind spot in principle.

[#336](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/336) measured the tree and left
two questions its measurement could not settle: which files the gate measures, and what it does
about definitions it cannot see. This ADR answers both.

Measured 2026-08-20 against the working tree at `6d7dd10cef`: 384 tracked `.jl` files and 4684
definitions.

Twelve macros are declared in `src/`. Five of them — `@fprop`, `@vprop`, `@pprop`, `@cprop` and
`@wprop` — prefix a field inside a struct body and declare nothing of their own. The other seven
each turn a declaration into definitions:

| macro | call sites | files |
| --- | --- | --- |
| `@propagatable` | 124 | 75 |
| `@forward_properties` | 22 | 21 |
| `@pipe_delegates` | 7 | 7 |
| `@windowed_estimator` | 6 | 5 |
| `@pipe_route_sigma_ucs` | 5 | 5 |
| `@pipe_route_rkb` | 2 | 2 |
| `@define_pretty_show` | 2 | 2 |

Two measurements bound the problem, and both shrink it.

**The parser already descends into a macro call's arguments.** `@propagatable struct … end`
encloses a hand-written inner constructor, and that constructor is measured normally:
`EntropyPoolingPrior` at `src/13_Prior/12_EntropyPoolingPrior.jl:1201` measures cyclomatic 16, and
the file's own measurement reports it, even though the `@propagatable` block opens at line 1144.
Removing every `@propagatable` prefix and re-measuring changes nothing at all — 3692 definitions
and 108 offending files, before and after, over all 196 files in scope. **No hand-written code
hides behind a macro.** The blind spot is confined to what a macro emits.

**What a macro emits is either trivial, or already measured at the macro itself.** The left column
is the macro's own measurement, at its declaration in `src/`. The right column is the worst that
its expansion *adds*, over every call site, beyond what the parser already sees.

| macro | declared cyc / cog / args | worst addition cyc / cog / args |
| --- | --- | --- |
| `@forward_properties` | 45 / 96 / 2 | 17 / 31 / 2 |
| `@define_pretty_show` | 28 / 54 / 2 | 27 / 43 / 2 |
| `@windowed_estimator` | 19 / 26 / 2 | 1 / 1 / **6** |
| `@propagatable` | 4 / 4 / 1 | 1 / 0 / 4 |
| `@pipe_delegates` | 2 / 4 / 2 | 1 / 1 / 3 |
| `@pipe_route_rkb` | 1 / 0 / 1 | 2 / 2 / 3 |
| `@pipe_route_sigma_ucs` | 1 / 0 / 1 | 1 / 0 / 3 |

Wherever the numbers are large enough to matter, the declaration exceeds the worst addition, so it
is already an offender in the baseline and already a candidate for the scheduled job. Where the
declaration does not bound the addition — `@pipe_route_rkb` declares 1 and adds 2 — both numbers
sit an order of magnitude below the thresholds, so the gap cannot hide an offender. Argument count
is the one exception, and it is handled below.

Only five files in `src/` measure a structural zero, and every one of them is a single
`@windowed_estimator` call. The other 39 zero-definition tracked files are test files, example
scripts and the module's include list, which hold no definitions to find.

Measuring expanded code was rejected. It costs a package load, it cannot reach a script, and it
pulls dependency boilerplate into the numbers: expanding
`src/20_Optimisation/13_NearOptimalCentering.jl` raises that file's cyclomatic maximum from 12 to
35, and the 35 is an `ArgCheck.@argcheck` branch pair wrapped around `Base.CoreLogging.@warn`'s
`let` machinery. A gate that counts them measures `ArgCheck` and `Logging`, not this library.

## Decision

### The measured scope is `src/` and `ext/`

196 files, 3692 definitions, 108 files over at least one threshold. `test/`, `docs/`, `examples/`,
`user_guide/` and `research/` are not measured.

This also retires the expectation that the two tools must disagree about scope. `report_package`
reaches the package's own modules and its loaded extensions, which is the same ground, so one
statement of scope now serves both halves of the gate.

### Every tracked `.jl` file is measured, or is a named Unmeasured Path

An **Unmeasured Path** is a committed entry naming a path the gate does not measure, carrying a
written reason. It has no baseline row, no number, and nothing to ratchet.

The gate asserts total coverage: every tracked `.jl` file must be either under a measured root or
matched by an Unmeasured Path. A tracked file that is neither turns the gate red. Five entries
cover the tree today — `test/`, `examples/`, `research/`, `user_guide/` and `docs/`.

The assertion is the whole point. A named root list on its own leaves a new directory unmeasured
and silent, which is the gap this decision exists to close. With the assertion, a new top-level
directory turns the gate red until someone rules on it, and the ruling is a commit.

`research/` carries its own reason: those files are the subject of the live prototypes-to-seams
effort, which will rewrite or delete many of them, and the scheduled job files each offending file
at most once ever. An issue against a file that is about to be deleted spends a permanent slot.

An Unmeasured Path is neither an Exemption nor a Dismissal, and the three must stay distinguishable
on sight:

| noun | what it says | what it binds |
| --- | --- | --- |
| **Unmeasured Path** | this path is never measured | nothing — no row exists |
| Exemption | this number is real and may stand | issue candidacy only |
| Dismissal | this JET report class is not a defect | the reviewed count |

### A Declaration Macro is measured at its declaration

A **Declaration Macro** is a macro that turns a declaration into definitions the parser cannot see
— the seven named above. The repo's own word is used deliberately: `generator` collides with
`@generated` functions and with generator expressions, both of which a reader of this codebase
meets often.

The gate does not expand. A Declaration Macro's complexity is measured once, where it is declared,
and that measurement stands for every site that calls it. This keeps the parser pure: no package
load, no inference, and no dependency boilerplate in the numbers.

### The Expansion Bound records what the declaration cannot

The **Expansion Bound** is a committed table with one row per (Declaration Macro, metric),
recording the worst that macro's expansion adds at any call site, beyond what the parser already
sees. Seven macros and three metrics — about 21 numbers, against the baseline's 3692 definitions.
It is seeded from the right-hand column of the table above.

It ratchets exactly as the baseline does, so it is green the day it lands and turns red when a
change to a Declaration Macro makes its expansion worse. Asserting instead that the declaration
bounds its own expansion was rejected: it starts red, because `@windowed_estimator` declares 2
arguments and emits methods taking 6 against a frozen threshold of 5.

Argument count is the metric where the declaration is not a proxy at all. The Expansion Bound
records the 6 and ratchets from it, so the breach is written down and a regression is caught, while
nothing turns red on the day the gate arrives.

### The Expansion Bound is remeasured when a Declaration Macro changes

Remeasuring needs the package loaded and every call site expanded — about 80 s, against the 4.8 s
the whole measurement costs. So it runs only when one of the four files that declare a Declaration
Macro changes: `src/01_Base.jl`, `src/02_Tools.jl`, `src/08_Moments/01_Base_Moments.jl` and
`src/20_Optimisation/01_Base_Optimisation.jl`. That fired on 41 of the last 76 commits.

Keying the trigger on the declaration's own baseline row would have been free, and it was rejected:
widening an emitted method's arity moves the expansion and leaves the declaration's own cyclomatic
number untouched, which is precisely the case the remeasure exists to catch.

### A baseline row names the Declaration Macros its file calls

A file whose measurement depends on a Declaration Macro carries a marker naming the macros it
calls, so a structural zero is never bare. The reader at risk is whoever reads the baseline or its
diff, and the fact travels with the number rather than living somewhere the reader must already
suspect. The baseline generator parses every file anyway, so detecting the calls costs nothing.

## Consequences

**`test/` is no longer measured.** That is 95 tracked files, and the loss is deliberate. Test code
was in the scope this map charted, and the narrower scope replaces it.

**The scheduled job's lifetime estimate survives.** 108 offending files in scope, against the
figure of about 110 that
[#342](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/342) reasoned from.

**A new top-level directory turns the gate red.** That is the cost of never being silently
unmeasured, and it is paid once per directory, by a commit that rules on it.

**[#353](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/353) inherited a sharper
question and settled it.** A file the baseline does not name is either an Unmeasured Path or a
coverage failure.
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md) makes the second
reading a set comparison, and it extends the same rule to the Expansion Bound's key set: a new
Declaration Macro's key is recorded green on arrival, because no threshold measures an addition.

**[#340](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/340) sites two more committed
files** — the Unmeasured Path list and the Expansion Bound — on top of the threshold configuration
and the exemption list that #342 handed it.

**The vocabulary stays out of `CONTEXT.md`.** That file's preamble scopes it to the library's
domain, and these nouns belong to the tooling, for the reason
[ADR 0071](0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md) gave.
