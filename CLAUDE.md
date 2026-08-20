# PortfolioOptimisers.jl — working agreements

A Julia portfolio-optimisation library. Estimators are configuration, Results are data, and the
domain vocabulary is normative — read `CONTEXT.md` before touching anything you cannot already name.

## Orientation

- **`STANDARDS.md` is the map.** It routes a subject — a docstring, an `export`, a constructor, a
  test — to the file that owns the rule and to the check that fails when the rule breaks. Start
  there when you do not know which file governs what you are about to change. It carries no rules
  of its own, so it never overrides the file it points at. `/improve-standards` audits it.
- **`CONTEXT.md` is the glossary.** It defines the domain language (Prior, Constraint Space,
  Universe Sets, Feature Matrix, Objective Penalty, …) and is hand-written. When you introduce or
  rename a concept, update it in the same change.
- **`docs/adr/` records decisions.** An ADR whose decision reached `main` is **amended, never
  rewritten**: append a `## Amendment (YYYY-MM-DD)` section. Such an ADR describes released
  behaviour, so an ADR describing superseded behaviour is correct history, not a bug. An ADR whose
  decision has **not** reached `main` is still a draft — rewrite it in place, because no reader
  outside the branch ever saw the text you would be amending.
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
- **JuliaFormatter can run over `src/`.** It escapes a quote inside a jldoctest block as `\"`.
  This is normal formatter output, not corruption: inside a `"""` docstring `\"` renders as `"`,
  and the doctests pass. Do not revert the escaping. `test/`, `examples/` and `user_guide/` are
  formatted in place too.
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
- **Never export an abstract type unless explicitly told to.** All but a handful of the abstract
  types in `src/` are unexported, so unexported is the convention. An open family, a sibling family
  that exports its supertype, and an existing API-page entry are none of them a reason to add one in
  passing — an export is public API, and widening it is the maintainer's call. Ask instead.
  `test/test_43_exported_abstract_type_census.jl` gates the rule against the allow-list in that
  file, so an export is a deliberate edit to that list. **Do not restate the count here or
  anywhere else.** It has moved four times, and each written copy went stale where it stood.

## Repo etiquette

- Never link to or post in repositories outside `dcelisgarza`'s — name external sources in prose.
- Branch before committing if you are on the default branch, and only commit when asked.
