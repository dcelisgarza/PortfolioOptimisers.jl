---
status: accepted
---

# A Returns Source is a singleton type

## Context

ADR [0211](0211-each-reader-of-the-returns-states-its-own-source.md) puts an `x_src` field on each
reader estimator of the returns, with the default `nothing`. The map "`x_src` takes over `brt`"
(#1678) moves the frame of `brt` into that field, so the excess returns `X - B` become one of its
values. This ADR decides the values, their type, their names, and the values that each reader admits
(#1681).

The facts at `1a8118bc13`:

- `x_src` is a `Symbol`, `:prior` or `:data`, checked by `assert_source_selector`. ADR
  [0044](0044-matrix-sources-are-named-not-flagged.md) chose a named `Symbol` over a `Bool`.
- `returns_matrix_picker(pr, rd, x_src, brt)` reads `pr.X` when `rd` is `nothing` or `x_src` is
  `:prior`, and otherwise the matrix that `benchmark_relative_returns(rd, brt)` gives.
- `resolve_factor_regression` takes `pr.rr` first, and refits only when the prior holds no loadings.
- ADR 0211 moves the field off the three heads and onto about 23 reader types, so the public API
  breaks whatever form the values take.
- The value sets differ by family: a prior cannot read a prior, and a graph reader and a regression
  reader can.
- In this library, a type whose name ends in `Returns` is mostly an estimator of expected returns,
  for example `ExcessExpectedReturns` (in excess of the risk-free rate) and `PriorExpectedReturns`.

## Decision

The maintainer decided on 2026-10-13.

1. **The values are singleton types.** An unexported abstract type `AbstractReturnsSource` has three
   exported subtypes. Dispatch on the value picks the matrix. The field keeps the name `x_src`, and
   its type is `Option{<:AbstractReturnsSource}`. The default stays `nothing`, which means "not
   stated" (ADR 0211).

   | Value | A prior or a graph reader reads | A regression reader takes |
   | --- | --- | --- |
   | `nothing` | `rd.X`. A graph reader reads `pr.X` when no `rd` is present. | `pr.rr`, else a refit on `rd.X` |
   | `DataSource()` | `rd.X` | a refit on `rd.X` |
   | `ExcessSource()` | `rd.X - rd.B`, the Excess Frame | a refit on `rd.X - rd.B` |
   | `PriorSource()` | `pr.X`, in the frame of the prior | `pr.rr` only |

2. **A stated raw value exists beside `nothing`.** `DataSource()` reads the same matrix as `nothing` on
   a prior and on a graph reader. It states that the caller chose the raw returns. So the frame check
   of #1682 can tell a reader that the caller forgot from a mix that the caller stated. On a
   regression reader, `DataSource()` differs from `nothing`: it always refits.
3. **One prior value.** `PriorSource()` reads `pr.X` as it is, so its frame is the frame that the prior
   fitted in. No raw form and no excess form of it exist. Under `x_src = :prior`, `brt` acts today only
   through the prior, so this keeps the pinned outputs of `test/test_78_brt_pinned_outputs.jl`.
4. **A stated value overrides the loadings of the prior.** On a regression reader, `DataSource()` and
   `ExcessSource()` always refit. `PriorSource()` takes `pr.rr` only, and throws an `ArgumentError`
   when the prior holds no loadings. The error tells the caller to set `nothing`, `DataSource()` or
   `ExcessSource()` to fit them on the data.
5. **The values that each reader admits.**

   | Reader | `nothing` | `DataSource()` | `ExcessSource()` | `PriorSource()` |
   | --- | --- | --- | --- | --- |
   | A prior estimator | yes | yes | yes | no |
   | A graph reader | yes | yes | yes | yes |
   | A regression reader | yes | yes | yes | yes |
   | `OnlinePortfolioSelection` | yes | yes | yes | no |

   A prior cannot read the returns of a prior. The step of `OnlinePortfolioSelection` reads the price
   relative of the returns data, so it has no prior to read.
6. **A constructor refuses a value that its reader does not admit.** All readers share the one field
   type. Each keyword constructor calls one shared assertion with its admitted set. The
   `ArgumentError` names the field, the reader, the value and the admitted set. The configuration
   alone decides the refusal, so it goes in the constructor.
7. **A stated source that the call cannot give warns, then falls back.** A graph reader can be called
   with a prior result and no returns data, or with a bare matrix. A stated `DataSource()` or
   `ExcessSource()` with no `rd`, and a stated `PriorSource()` on a bare matrix, give a warning. Then
   the reader reads `pr.X`, or the matrix. `nothing` falls back with no warning.
8. **Unchanged rules.** `ExcessSource()` on returns data that carries no benchmark throws, as
   `benchmark_relative_returns` does today. `x_src` stays static, with no `TimeDependent` schedule
   (ADR 0044).

## Consequences

- `assert_source_selector` and its `Symbol` check go away. The shared assertion of decision 6 takes
  their place.
- `returns_matrix_picker` and `benchmark_relative_returns` lose their `Symbol` and `Bool` arguments.
  One method per value of `AbstractReturnsSource` picks the matrix.
- The step of `OnlinePortfolioSelection` renames `brt` to `x_src` (ADR 0211). `head_frame_returns`
  reads the value of the step in place of `step.brt`.
- Each printed estimator gains an `x_src` row that prints `nothing` or a singleton.
- The rule of ADR 0044 holds: a source is named, not flagged. A singleton type is a name.
- Three exported names are added: `DataSource`, `ExcessSource` and `PriorSource`.

## Alternatives considered

- **A `Symbol` with `Val` dispatch**, as on #1494. That rule keeps an existing public keyword. Here
  ADR 0211 moves the field to new owners, so no API stays to keep. Each family also needs a runtime
  check of its value set, which a singleton type gives through dispatch.
- **No raw value, so `nothing` is raw.** Fewer types, but the frame check cannot tell a forgotten
  reader from a stated one.
- **A raw and an excess form of the prior value.** The second form must add or subtract `B` on the rows
  of `pr.X`. A factor prior reconstructs `pr.X`, and a scenario window and a gap cut its rows, so the
  rows of `B` need an alignment. It also mixes two frames in one value.
- **`PriorSource()` on a regression reader refits on `pr.X` when the prior holds no loadings.** It
  needs a row alignment of the factor returns to `pr.X`.
- **No `PriorSource()` on a regression reader.** The loadings of the prior are then reachable only
  through `nothing`, which falls back to a refit with no error.
- **A type split**, `AbstractDataSource <: AbstractReturnsSource`, so that the field type of a prior
  admits only the data values. The refusal is then a `MethodError` from the constructor that
  `@concrete` writes, and that message does not name the admitted values.
- **A stated source that the call cannot give throws.** The maintainer chose a warning.
- **Ignore a stated source with no `rd`**, as today. A stated source then has no effect with no sign.
- **The names `DataReturns`, `ExcessReturns` and `PriorReturns`.** They read as estimators of expected
  returns, and "excess" reads as the excess of the risk-free rate.
- **The name `BenchmarkExcessSource`.** It names the benchmark, but the glossary term Excess Frame
  already fixes the meaning of "excess".
- **Rename the field to `src` or `rsrc`.** The frame is a property of the matrix `X - B`, so "which
  returns matrix `X`" still describes the field.

## Related

- ADR 0044: the rule that named `x_src`.
- ADR 0204: the current rule for `brt`.
- ADR 0208: `DataTracking`, which refuses the Excess Frame.
- ADR 0211: the readers that hold `x_src`.
- `returns_matrix_picker`, `benchmark_relative_returns`, `resolve_factor_regression`,
  `head_frame_returns`, `assert_source_selector`.
