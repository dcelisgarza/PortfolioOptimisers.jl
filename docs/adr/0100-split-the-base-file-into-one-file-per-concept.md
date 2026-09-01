---
status: accepted
---

# Split the base file into one file per concept

## Context

`src/01_Base.jl` was 5495 lines and held 165 top-level bindings under no section marker. The
architecture review of 2026-09-01 counted about eight concepts in it and recorded the file as
real friction that it did not recommend acting on, because the size predates PR 625 and a split
is cross-cutting.

The concepts were these, and none of them needs any of the others to be in the same file:

- the seven docstring dictionaries and the `unique_key_dict` helper that builds them,
- the type roots `AbstractEstimator`, `AbstractAlgorithm`, `AbstractResult` and
  `DynamicAbstractWeights`, with the length-1 iteration protocol over them,
- the `@define_pretty_show` macro and the three `pretty_show_vector_*` helpers,
- the `ScopedConfig` holder and its four instances,
- the load-time preference channel and `__init__`,
- the message builders,
- the `PortfolioOptimisersError` hierarchy,
- about fifty type aliases,
- the observation-weights seam,
- the `assert_*` family, `resolve_rng` and `dims_oriented`,
- `VecScalar`,
- the `NormError` family,
- the Kaniadakis logarithm.

A reader after one of them read a table of contents that did not exist. Every gate that this
repository keys on a file — the sweep manifest, the three code-health baselines, the coverage
ratchet — held one row for all thirteen, so no number could say which concept carried the debt.
The `@define_pretty_show` macro drove the whole file's complexity maximum at cyc = 27 and
cog = 50, and `apply_preferences!` at cyc = 26 and cog = 33 stood under it, invisible.

## Decision

**One file per concept, in a `src/01_Base/` directory, in load order.**

| File | What it holds |
| ---- | ------------- |
| `01_DocstringDictionaries.jl` | `unique_key_dict` and the seven dictionaries |
| `02_TypeRoots.jl` | the four abstract roots and the length-1 iteration protocol |
| `03_PrettyShow.jl` | `@define_pretty_show`, `has_pretty_show_method`, the vector helpers |
| `04_ScopedConfig.jl` | `ScopedConfig` and its four instances with their setters |
| `05_Preferences.jl` | `PREFERENCE_KEYS`, `apply_preferences!`, `__init__` |
| `06_Messages.jl` | `did_you_mean` and the message builders |
| `07_Errors.jl` | the `PortfolioOptimisersError` hierarchy and `showerror` |
| `08_TypeAliases.jl` | the type aliases and the two value-algorithm roots |
| `09_ObservationWeights.jl` | `ObsWeights` and `get_observation_weights` |
| `10_Assertions.jl` | the `assert_*` family, `resolve_rng`, `dims_oriented` |
| `11_VecScalar.jl` | `VecScalar` and the two aliases that name it |
| `12_NormError.jl` | the `NormError` family, `norm_factor`, `norm_error` |
| `13_KappaLog.jl` | `kappa_log` |

The order is the order the file already had, with two blocks moved to the concept they belong
to: the iteration protocol joins the type roots it iterates, and the `pretty_show_vector_*`
helpers join the macro that calls them. No other line moves relative to another.

**The move is structural.** The moved text is byte-for-byte the text that stood in the base
file. No type, no bound, no verb and no docstring changed. Three numbers prove it: the thirteen
files carry 139 documented units, 40 `# Algorithm` sections and 423 relevant lines, which are
the three numbers the base file's rows recorded.

**A new base symbol goes in the file named for its concept.** A concept that no file names gets
a new file rather than a section in an existing one. This is what stops the directory becoming
the file it replaced.

### What follows the split

- `src/PortfolioOptimisers.jl` includes the thirteen files in place of the one.
- The `export` list splits three ways: the error names to `07_Errors.jl`, `VecScalar` to
  `11_VecScalar.jl`, and the five norm types to `12_NormError.jl`.
- `sweep/manifest.toml` takes thirteen rows at `map = 1`, `swept = true`, whose units sum to the
  139 the old row carried. The sweep of #439 read this text and it did not change.
- The complexity, JET and coverage baselines take thirteen rows each. The coverage rows split
  the 423 lines of the lcov of run 33485210033 by line range, and every one of them enters
  terminal at zero misses under ADR 0082.
- `code_health/rulings.toml` takes four Exemptions. `@define_pretty_show` and
  `apply_preferences!` each now drive their own file's maximum, and ADR 0074's entry test reads
  a moved definition as a new one. Neither body changed.
- `CodeHealth.DECLARING_FILES` names `src/01_Base/03_PrettyShow.jl`, which declares the macro.
- The API page `docs/src/api/01_Base.md` stays one page. A page is checked by the units it
  renders, never by the file they are declared in, and `docs/make.jl` indexes the API
  subdirectories positionally.

## Consequences

A reader after one concept opens one file, and a file's name says what is in it. A gate's number
now names a concept: the two definitions above a complexity threshold stand alone in their own
rows rather than hidden inside a file maximum.

The directory is a new shape for a top-level `src/` entry that was a file. A reader who knows the
old path finds nothing at it. Every reference under `src/`, `test/`, `code_health/`, `CLAUDE.md`,
`STANDARDS.md`, `.github/` and `docs/` moves with the split. An ADR that names the old path in
prose keeps it: such an ADR describes released behaviour, and the statement was true when it was
written.

## Alternatives considered

- **Section markers in one file.** A comment banner is not a boundary. Nothing measures it, no
  gate keys on it, and the file keeps one row for every concept in it.
- **Fewer, larger files.** Three or four files grouping related concepts would leave the same
  question the split answers: which concept owns the number the row carries.
- **Keep `src/01_Base.jl` as a file of includes.** It would hold every link that names the old
  path, at the cost of an include level that no other file in `src/` has, and of a row in every
  baseline that measures nothing.
