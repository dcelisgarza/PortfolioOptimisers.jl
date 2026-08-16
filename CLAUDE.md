# PortfolioOptimisers.jl — working agreements

A Julia portfolio-optimisation library. Estimators are configuration, Results are data, and the
domain vocabulary is normative — read `CONTEXT.md` before touching anything you cannot already name.

## Orientation

- **`CONTEXT.md` is the glossary.** It defines the domain language (Prior, Constraint Space,
  Universe Sets, Feature Matrix, Objective Penalty, …) and is hand-written. When you introduce or
  rename a concept, update it in the same change.
- **`docs/adr/` records decisions.** ADRs are **amended, never rewritten**: append a
  `## Amendment (YYYY-MM-DD)` section. An ADR describing superseded behaviour is correct history,
  not a bug.
- To find code, prefer kaimon's `search_code` (semantic) when you can only *describe* what you
  want, and `grep_code` when you already hold an exact token. `/graphify` builds a queryable graph
  for larger architectural questions.

## Running Julia

- **Go through the kaimon MCP tools**, not `julia` on Bash. `ex(e="…")` evaluates in a REPL the
  user shares live.
- **Single-threaded**: `julia -t 1`, `BLAS.set_num_threads(1)`. Never kick off the full test suite
  or a docs build — those are the maintainer's to run.
- Run **targeted** `test_*.jl` files for the area you changed. `test/runtests.jl` supplies a shared
  `init_code` preamble (`Test`, `Logging`, `CSV`, `TimeSeries`, `DataFrames`, `StableRNGs`,
  `StatsBase`, `LinearAlgebra`, `find_tol`); reproduce it if you include a test file directly.
- On a Revise world-age warning, **restart the session and cold-load**. If a session wedges, shut it
  down and start a new one rather than fighting it.

## Editing

- **Only edit `.jl`.** These are generated and will be overwritten by CI:
  `examples/**/*.ipynb`, `docs/src/examples/**`, `docs/src/user_guide/**`,
  `docs/src/capability_catalogue.md`, `docs/src/api/*_TypeHierarchy.md`.
  Their sources are `examples/**/*.jl`, `user_guide/*.jl`, `docs/capability_catalogue.jl`.
  `docs/src/api/**` (except the type hierarchy) is hand-written.
- **Do not run JuliaFormatter directly over `src/`** — it has corrupted escaped quotes inside
  jldoctest blocks. Safe pattern when a change needs reflowing: copy the changed files to a scratch
  directory with `.JuliaFormatter.toml`, format *there*, diff, and copy back only if the diff is
  clean. `test/`, `examples/` and `user_guide/` can be formatted in place.
- Margin is 92 (`.JuliaFormatter.toml`, `yas` style). Long string literals and docstring prose are
  exempt in practice; code lines are not.

## Doctests

- See the `run-doctests` skill for the fresh-process rule and the exact CI invocation to mirror.

## Design rules

- **Estimators never hold Results internally.** Enforce it with the field's *type bound*, not a
  runtime check; precomputed structure belongs on the Result type.
- **Prefer a per-type method over a new dependency** for reflection-style work. Derive the field
  list, write the constructor name once per type, and use the ordinary keyword constructor.
- Docstring field text is centralised in `field_dict` / `arg_dict` in `src/01_Base.jl`. Add an entry
  there rather than inlining prose, and delete entries that lose their last user.
- **Never export an abstract type unless explicitly told to.** 221 of 226 abstract types in `src/`
  are unexported, so unexported is the convention. An open family, a sibling family that exports its
  supertype, and an existing API-page entry are none of them a reason to add one in passing — an
  export is public API, and widening it is the maintainer's call. Ask instead.

## Repo etiquette

- Never link to or post in repositories outside `dcelisgarza`'s — name external sources in prose.
- Branch before committing if you are on the default branch, and only commit when asked.
